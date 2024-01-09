from django.contrib import admin
from django.urls import path

import djangoProjectMLLab
from djangoProjectMLLab.views import upload_file

urlpatterns = [
    path('admin/', admin.site.urls),
    path('',upload_file)
]

