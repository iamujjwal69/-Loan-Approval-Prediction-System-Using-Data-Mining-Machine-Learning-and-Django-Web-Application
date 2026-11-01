"""
Database Models for Datasets App.
Stores metadata, file references, and statistical metrics of uploaded datasets.
"""

from django.contrib.auth.models import User
from django.db import models


class Dataset(models.Model):
    STATUS_CHOICES = [
        ("UPLOADED", "Uploaded"),
        ("PROCESSED", "Processed"),
        ("INVALID", "Invalid"),
    ]

    FILE_TYPE_CHOICES = [
        ("CSV", "CSV File"),
        ("XLSX", "Excel Spreadsheet"),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="datasets")
    dataset_name = models.CharField(max_length=255)
    file = models.FileField(upload_to="uploaded_datasets/")
    file_type = models.CharField(max_length=10, choices=FILE_TYPE_CHOICES, default="CSV")
    total_records = models.IntegerField(default=0)
    total_columns = models.IntegerField(default=0)
    missing_values_count = models.IntegerField(default=0)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="UPLOADED")
    summary_metadata = models.JSONField(default=dict, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-uploaded_at"]

    def __str__(self):
        return f"{self.dataset_name} ({self.total_records} records) - {self.status}"
