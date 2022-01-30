from django import forms
from main.models import FinancialBackend


class FinancialBackendForm(forms.ModelForm):
    symbol = forms.CharField(max_length=8, help_text='Please provide Company Symbol!')

    class Meta:
        model = FinancialBackend
        fields = ('symbol',)
