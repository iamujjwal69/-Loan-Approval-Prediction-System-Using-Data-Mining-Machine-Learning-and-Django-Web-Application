from django.urls import path
from analytics import views

urlpatterns = [
    path("", views.analytics_dashboard_view, name="analytics_dashboard"),
    path("api/charts/", views.api_chart_data, name="api_chart_data"),
]
