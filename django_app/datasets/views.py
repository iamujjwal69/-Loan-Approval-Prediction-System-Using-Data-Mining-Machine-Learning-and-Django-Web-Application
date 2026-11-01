"""
Views for Datasets App: Upload, Preview, Preprocessing Inspection, and Clean Export.
"""

import io
from pathlib import Path
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
import pandas as pd

from datasets.forms import DatasetUploadForm
from datasets.models import Dataset
from src.config import MODEL_FEATURES, RAW_DATA_PATH
from src.data_cleaning import (
    clean_categorical_values,
    get_missing_summary,
    impute_missing_exploratory,
    remove_duplicates,
)
from src.data_loader import get_dataset_overview, load_dataset, validate_dataset_schema
from src.outlier_detection import cap_outliers_iqr, get_all_outliers_summary


@login_required
def upload_dataset_view(request):
    if request.method == "POST":
        form = DatasetUploadForm(request.POST, request.FILES)
        if form.is_valid():
            dataset_instance = form.save(commit=False)
            dataset_instance.user = request.user

            file_ext = Path(dataset_instance.file.name).suffix.lower()
            dataset_instance.file_type = "XLSX" if file_ext in [".xlsx", ".xls"] else "CSV"
            dataset_instance.save()

            # Inspect and validate uploaded content
            try:
                df = load_dataset(dataset_instance.file.path)
                is_valid, validation_errors = validate_dataset_schema(df, require_target=False)

                if not is_valid:
                    dataset_instance.status = "INVALID"
                    messages.warning(
                        request,
                        f"Dataset uploaded with schema warnings: {'; '.join(validation_errors)}",
                    )
                else:
                    dataset_instance.status = "UPLOADED"
                    messages.success(
                        request,
                        f"Dataset '{dataset_instance.dataset_name}' uploaded and verified successfully!",
                    )

                overview = get_dataset_overview(df)
                dataset_instance.total_records = overview["total_records"]
                dataset_instance.total_columns = overview["total_columns"]
                dataset_instance.missing_values_count = overview["total_missing_cells"]
                dataset_instance.summary_metadata = overview
                dataset_instance.save()

                return redirect("dataset_preview", dataset_id=dataset_instance.id)

            except Exception as e:
                dataset_instance.status = "INVALID"
                dataset_instance.save()
                messages.error(request, f"Error processing file: {str(e)}")
                return redirect("dataset_upload")
        else:
            messages.error(request, "Failed to upload dataset. Check form constraints.")
    else:
        form = DatasetUploadForm()

    recent_datasets = Dataset.objects.filter(user=request.user)[:5]
    return render(
        request,
        "datasets/upload.html",
        {"form": form, "recent_datasets": recent_datasets},
    )


@login_required
def dataset_preview_view(request, dataset_id=None):
    if dataset_id:
        dataset = get_object_or_404(Dataset, id=dataset_id)
        file_path = dataset.file.path
    else:
        dataset = Dataset.objects.first()
        file_path = RAW_DATA_PATH

    try:
        df = load_dataset(file_path)
    except Exception:
        df = load_dataset(RAW_DATA_PATH)
        dataset = None

    overview = get_dataset_overview(df)

    # Paginate DataFrame records (15 per page)
    records = df.where(pd.notnull(df), None).to_dict(orient="records")
    paginator = Paginator(records, 15)
    page_number = request.GET.get("page", 1)
    page_obj = paginator.get_page(page_number)

    all_datasets = Dataset.objects.all()

    context = {
        "dataset": dataset,
        "overview": overview,
        "page_obj": page_obj,
        "columns": list(df.columns),
        "all_datasets": all_datasets,
    }
    return render(request, "datasets/preview.html", context)


@login_required
def dataset_preprocessing_view(request, dataset_id=None):
    if dataset_id:
        dataset = get_object_or_404(Dataset, id=dataset_id)
        file_path = dataset.file.path
    else:
        dataset = Dataset.objects.first()
        file_path = RAW_DATA_PATH

    try:
        raw_df = load_dataset(file_path)
    except Exception:
        raw_df = load_dataset(RAW_DATA_PATH)
        dataset = None

    # Before stats
    before_missing = get_missing_summary(raw_df).to_dict(orient="records")
    outlier_summary = get_all_outliers_summary(raw_df).to_dict(orient="records")

    # Apply data mining transformations
    cleaned_df, dupes_removed = remove_duplicates(raw_df)
    cleaned_df = clean_categorical_values(cleaned_df)
    capped_df, bounds_applied = cap_outliers_iqr(cleaned_df)
    imputed_df, filled_values = impute_missing_exploratory(capped_df)

    # After stats
    after_missing = get_missing_summary(imputed_df).to_dict(orient="records")

    context = {
        "dataset": dataset,
        "before_missing": before_missing,
        "after_missing": after_missing,
        "outlier_summary": outlier_summary,
        "bounds_applied": bounds_applied,
        "filled_values": filled_values,
        "dupes_removed": dupes_removed,
        "total_records_before": len(raw_df),
        "total_records_after": len(imputed_df),
        "dataset_id": dataset.id if dataset else 0,
    }
    return render(request, "datasets/preprocessing.html", context)


@login_required
def download_processed_csv_view(request, dataset_id=None):
    if dataset_id and dataset_id != 0:
        dataset = get_object_or_404(Dataset, id=dataset_id)
        file_path = dataset.file.path
    else:
        file_path = RAW_DATA_PATH

    raw_df = load_dataset(file_path)
    cleaned_df, _ = remove_duplicates(raw_df)
    cleaned_df = clean_categorical_values(cleaned_df)
    capped_df, _ = cap_outliers_iqr(cleaned_df)
    imputed_df, _ = impute_missing_exploratory(capped_df)

    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = 'attachment; filename="preprocessed_loan_data.csv"'
    imputed_df.to_csv(path_or_buf=response, index=False)
    return response
