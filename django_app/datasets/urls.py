from django.urls import path
from datasets import views

urlpatterns = [
    path("upload/", views.upload_dataset_view, name="dataset_upload"),
    path("preview/", views.dataset_preview_view, name="dataset_preview_default"),
    path("preview/<int:dataset_id>/", views.dataset_preview_view, name="dataset_preview"),
    path("preprocessing/", views.dataset_preprocessing_view, name="dataset_preprocessing_default"),
    path("preprocessing/<int:dataset_id>/", views.dataset_preprocessing_view, name="dataset_preprocessing"),
    path("download-clean/<int:dataset_id>/", views.download_processed_csv_view, name="dataset_download_clean"),
]
