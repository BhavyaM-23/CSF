from django.shortcuts import render,HttpResponse
from . import predict
from . models import uploadfile
import os

# our home page view
def home(request):    
    return render(request, 'index.html')
        
def save_files(request):
    if request.method == "POST":
        name=request.POST.get("patient")
        di=request.FILES.getlist("files")
        uploadfile.objects.all().delete()
        for f in di:
            uploadfile(patient=name, files=f,).save()
        return result(request)

# our result page view
def result(request):
    result = predict.pred()
    print(result)
    s=""
    if result[7]>0.5:
        s+="** FRACTURED **"
        ind=result.index(max(result[:7]))
        s+="\nPredicted fracture at C"+str(ind+1)
    else:
        s+="** NOT FRACTURED **"

    return HttpResponse(s)