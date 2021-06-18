from django.shortcuts import render

def ru(request):
    content = {}
    return render(request, 'open/ru.html', content)

def terms(request):
    content = {}
    return render(request, 'open/terms.html', content)
