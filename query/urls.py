from . import views
from django.urls import path

urlpatterns = [
    path('load_vectors/', views.load_vectors, name='load_vectors'),
    # path('query/', views.query, name='query'),
    path('', views.query, name='query'),
    path('query_result/<str:query_request>', views.query_result, name='query_result'),
]