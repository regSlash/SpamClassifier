from django.urls import path
from api.views import SpamClassificationView

urlpatterns = [
    path('classify/', SpamClassificationView.as_view(), name='classify'),
]