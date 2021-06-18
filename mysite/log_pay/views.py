from django.shortcuts import render
from django.contrib.admin.models import LogEntry
from django.template import loader
from django.http import HttpResponse, HttpResponseRedirect
from .models import *
from django.http import Http404
from django.shortcuts import get_object_or_404, render
from django.urls import reverse
from django.views import generic
from django.utils import timezone
from django.contrib.auth.decorators import login_required
import datetime
from django.core.serializers import serialize
from .forms import *
from django.contrib.auth.models import User
from django.contrib.auth import logout
import time
from django.views.generic import View
import plotly.offline as opy
import plotly.graph_objs as go
from plotly.offline import plot
import urllib.request
import re
import pandas as pd
import numpy as np
import json
import requests

now = datetime.date.today()
def custom_logout(request):
    print('Loggin out {}'.format(request.user))
    logout(request)
    print(request.user)
    return HttpResponseRedirect('')

    
@login_required
def faq(request):
    vessels = Vessel.objects.all()
    cats = Type_cat.objects.all()
    content = {'vessels':vessels, 'TYPE_VESSEL':TYPE_VESSEL, 'cats':cats}
    return render(request, 'log_pay/faq.html', content)

def registration(request):
    if request.POST:
        username = request.POST['name']
        email = request.POST['email']
        password = request.POST['password']
        password2 = request.POST['password2']
        if password == password2:
            new_user = User.objects.create_user(username, email, password)
            new_user.save()
        return HttpResponseRedirect('/accounts/login/')
    content = {}
    return render(request, 'registration/registration.html', content)

def password_reset(request, is_admin_site=False):
    if request.POST:
        return HttpResponseRedirect('/accounts/login/')
    content = {}
    return render(request, 'registration/password_reset.html', content)

class LogoutView(View):
    template_name = 'registration/logged_out.html'

    def get(self, request):
        response = logout(request)

        return render(response, self.template_name)
    
@login_required
def index(request):
    
    content = {}
    return render(request, 'log_pay/index.html', content)
