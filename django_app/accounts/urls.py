from django.urls import path
from accounts import views

urlpatterns = [
    path("login/", views.user_login_view, name="login"),
    path("register/", views.user_register_view, name="register"),
    path("logout/", views.user_logout_view, name="logout"),
]
