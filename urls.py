from django.urls import path
from . import views

urlpatterns = [
    path('infer/', views.inference_api, name='inference_api'),
]