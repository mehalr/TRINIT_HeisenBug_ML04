from django.db import models

# Create your models here.


class FinancialBackend(models.Model):

    symbol = models.CharField(max_length=8, unique=True)
    model_path = models.CharField(max_length=512, blank=True)
    latest_prediction = models.CharField(max_length=12, blank=True)
    latest_prediction_date = models.DateField(auto_now=True)

    def __str__(self):
        return self.symbol

    def get_model_path(self):
        return self.model_path
