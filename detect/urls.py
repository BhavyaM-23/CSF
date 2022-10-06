from django.urls import path
from . import views

app_name = "detect"

urlpatterns = [
    path('', views.home, name='home'),
    path('upload/',views.save_files,name="upload"),
    path('result/', views.result, name='result')
]