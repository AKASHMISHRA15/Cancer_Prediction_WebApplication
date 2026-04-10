from django.urls import path
from . import views

urlpatterns = [
    path('', views.info, name='info'),  # This is now the homepage
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('profile/', views.profile_view, name='profile'),
    path('lung_cancer_checkup/', views.lung_cancer_checkup_view, name='lung_cancer_checkup'),
    path('breast_cancer_checkup/', views.breast_cancer_checkup, name='breast_cancer_checkup'),
   

]



