from django.urls import path
from predictions import views

urlpatterns = [
    path("", views.predict_loan_view, name="predict_loan"),
    path("history/", views.prediction_history_view, name="prediction_history"),
    path("models/", views.models_overview_view, name="models_overview"),
    path("export-history/", views.export_predictions_csv, name="export_predictions_csv"),
]
