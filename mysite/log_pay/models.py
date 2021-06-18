#-*- coding: utf-8 -*-
from django.core.files.storage import FileSystemStorage
from django.db import models
from django.contrib.auth.models import AbstractUser
import datetime
from django.utils import timezone
from django.conf import settings
from datetime import datetime
from django.contrib.auth.models import User
# Create your models here.
private_storage = FileSystemStorage(location=settings.PRIVATE_STORAGE_ROOT)
media_storage = FileSystemStorage(location=settings.MEDIA_ROOT)
