"""
Database Models for Predictions & Model Governance App.
Stores historical model benchmarks and audited loan applicant predictions.
"""

from django.contrib.auth.models import User
from django.db import models


class ModelPerformance(models.Model):
    model_name = models.CharField(max_length=100)
    accuracy = models.FloatField()
    precision = models.FloatField()
    recall = models.FloatField()
    f1_score = models.FloatField()
    roc_auc = models.FloatField(null=True, blank=True)
    cv_mean_accuracy = models.FloatField(default=0.0)
    confusion_matrix = models.JSONField(default=dict, blank=True)
    classification_report = models.JSONField(default=dict, blank=True)
    is_active_model = models.BooleanField(default=False)
    hyperparameters = models.JSONField(default=dict, blank=True)
    trained_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-f1_score", "-accuracy"]

    def __str__(self):
        active_str = " [ACTIVE]" if self.is_active_model else ""
        return f"{self.model_name}{active_str} (Acc: {self.accuracy:.2%}, F1: {self.f1_score:.2%})"


class PredictionHistory(models.Model):
    STATUS_CHOICES = [
        ("Y", "Approved"),
        ("N", "Rejected"),
    ]

    user = models.ForeignKey(
        User, on_delete=models.SET_NULL, null=True, blank=True, related_name="predictions"
    )
    model_used = models.ForeignKey(
        ModelPerformance,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="predictions",
    )
    applicant_data = models.JSONField()
    predicted_result = models.CharField(max_length=5, choices=STATUS_CHOICES)
    status_label = models.CharField(max_length=20, default="Approved")
    probability_approved = models.FloatField()
    probability_rejected = models.FloatField()
    risk_level = models.CharField(max_length=30, default="Low Risk")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"Prediction #{self.id} - {self.status_label} ({self.probability_approved:.1f}%) on {self.created_at.strftime('%Y-%m-%d %H:%M')}"
