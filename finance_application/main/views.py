from main.models import FinancialBackend
from main.forms import FinancialBackendForm
from main.stock_model import *
from django.shortcuts import render, redirect
import joblib
import keras

import os
from finance.settings import BASE_DIR


def stock_view(request, *args, **kwargs):
    sym = kwargs.get('symbol')
    stock = FinancialBackend.objects.get(symbol=sym)
    print(stock)
    return stock


def stock_create(request, *args, **kwargs):
    print(request.POST)
    if request.method == 'POST':
        # stock_form = FinancialBackendForm(request.POST)
        # if stock_form.is_valid():
        #     stock = stock_form.save()
        symbol = request.POST['symbol'][0]
        data = get_stock_data(symbol)
        df = create_dataframe(data)
        predict, loss = LSTM_ALGO(df, symbol)
        context = {}
        context['predict'] = predict
        context['loss'] = loss
        return render(request, 'new.html', context)
    else:
        return False


def stock_render(request):
    return render(request, 'front/index.html')


def stock_predict(request):
    print(request.POST)
    if request.method == 'POST':
        opened = request.POST['open']
        high = request.POST['high']
        low = request.POST['low']
        volume = request.POST['volume']

        loaded_1 = keras.models.load_model(os.path.join(BASE_DIR,'/main/mymodel'))
        my_scaler = joblib.load('my_dope_model.pkl')
        v = my_scaler.transform([[134],[float(opened)],[float(high)],[float(low)],[float(volume)]])
        prediction = loaded_1.predict(v)
        final_prediction = my_scaler.inverse_transform(prediction)

        print(final_prediction)
        return redirect('render')




