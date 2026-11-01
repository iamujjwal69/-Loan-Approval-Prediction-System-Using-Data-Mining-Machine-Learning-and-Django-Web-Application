"""
Forms for Dataset Upload & Validation.
"""

from pathlib import Path
from django import forms
from datasets.models import Dataset

MAX_UPLOAD_SIZE_MB = 10


class DatasetUploadForm(forms.ModelForm):
    class Meta:
        model = Dataset
        fields = ["dataset_name", "file"]
        widgets = {
            "dataset_name": forms.TextInput(
                attrs={
                    "class": "form-control",
                    "placeholder": "e.g., Q3 Applicant Credit Dataset",
                }
            ),
            "file": forms.FileInput(
                attrs={
                    "class": "form-control",
                    "accept": ".csv, .xlsx, .xls",
                }
            ),
        }

    def clean_file(self):
        uploaded_file = self.cleaned_data.get("file")
        if not uploaded_file:
            raise forms.ValidationError("Please choose a file to upload.")

        # Extension check
        suffix = Path(uploaded_file.name).suffix.lower()
        if suffix not in [".csv", ".xlsx", ".xls"]:
            raise forms.ValidationError(
                f"Unsupported file format '{suffix}'. Please upload a valid .csv, .xlsx, or .xls file."
            )

        # Size check
        if uploaded_file.size > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
            raise forms.ValidationError(
                f"File size exceeds maximum allowed limit of {MAX_UPLOAD_SIZE_MB} MB."
            )

        return uploaded_file
