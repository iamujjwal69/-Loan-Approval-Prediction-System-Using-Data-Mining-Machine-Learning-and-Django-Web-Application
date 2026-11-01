"""
Views for Predictions App: Real-Time Inference, History Audit, Models Benchmarks, and CSV Export.
"""

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, render
import pandas as pd

from predictions.forms import ApplicantLoanForm
from predictions.models import ModelPerformance, PredictionHistory
from src.prediction import predict_loan_approval


@login_required
def predict_loan_view(request):
    prediction_result = None
    form = ApplicantLoanForm(request.POST or None)

    if request.method == "POST":
        if form.is_valid():
            applicant_data = form.to_applicant_dict()
            try:
                # Perform real-time inference using the persisted ML pipeline
                result = predict_loan_approval(applicant_data)
                prediction_result = result

                active_model = ModelPerformance.objects.filter(is_active_model=True).first()

                # Save record to database audit trail
                history_record = PredictionHistory.objects.create(
                    user=request.user,
                    model_used=active_model,
                    applicant_data=applicant_data,
                    predicted_result=result["predicted_result"],
                    status_label=result["status_label"],
                    probability_approved=result["probability_approved"],
                    probability_rejected=result["probability_rejected"],
                    risk_level=result["risk_level"],
                )
                result["record_id"] = history_record.id

                if result["predicted_result"] == "Y":
                    messages.success(request, f"Application #{history_record.id} assessed: LIKELY APPROVED ({result['probability_approved']}%)")
                else:
                    messages.warning(request, f"Application #{history_record.id} assessed: LIKELY REJECTED ({result['probability_rejected']}%)")

            except Exception as e:
                messages.error(request, f"Inference engine error: {str(e)}")
        else:
            messages.error(request, "Please check the form inputs for errors.")

    return render(
        request,
        "predictions/predict.html",
        {"form": form, "result": prediction_result},
    )


@login_required
def prediction_history_view(request):
    queryset = PredictionHistory.objects.select_related("user", "model_used").all()

    # Optional filtering by status or risk
    status_filter = request.GET.get("status")
    if status_filter in ["Y", "N"]:
        queryset = queryset.filter(predicted_result=status_filter)

    risk_filter = request.GET.get("risk")
    if risk_filter:
        queryset = queryset.filter(risk_level__icontains=risk_filter)

    paginator = Paginator(queryset, 10)
    page_number = request.GET.get("page", 1)
    page_obj = paginator.get_page(page_number)

    return render(
        request,
        "predictions/history.html",
        {
            "page_obj": page_obj,
            "status_filter": status_filter,
            "risk_filter": risk_filter,
            "total_count": queryset.count(),
        },
    )


@login_required
def models_overview_view(request):
    models = ModelPerformance.objects.all().order_by("-is_active_model", "-f1_score")
    active_model = models.filter(is_active_model=True).first()
    return render(
        request,
        "predictions/models_overview.html",
        {"models": models, "active_model": active_model},
    )


@login_required
def export_predictions_csv(request):
    records = PredictionHistory.objects.all().order_by("-created_at")
    
    rows = []
    for r in records:
        data = dict(r.applicant_data)
        data["Prediction_ID"] = r.id
        data["Predicted_Status"] = r.predicted_result
        data["Probability_Approved"] = f"{r.probability_approved:.1f}%"
        data["Risk_Level"] = r.risk_level
        data["Assessed_By"] = r.user.username if r.user else "System"
        data["Timestamp"] = r.created_at.strftime("%Y-%m-%d %H:%M:%S")
        rows.append(data)

    df = pd.DataFrame(rows)
    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = 'attachment; filename="loan_prediction_audit_history.csv"'
    df.to_csv(path_or_buf=response, index=False)
    return response
