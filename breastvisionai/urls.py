"""
URL configuration for breastvisionai project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.urls import path, include, re_path
from django.conf import settings
from django.conf.urls.static import static
from django.views.static import serve
from .views import frontend, root, UI_DIST

urlpatterns = [
    path('', root, name='root'),
    path('api/', include('api.urls')),
    path('api/', include('prediction.urls')),
    re_path(r'^assets/(?P<path>.*)$', serve, {'document_root': UI_DIST / 'assets'}),
    re_path(r'^(?P<path>logo\.png|hero-scan\.png|icons\.svg|favicon\.svg)$', serve, {'document_root': UI_DIST}),
    re_path(r'^(?:login|dashboard|diagnosis(?:/.*)?|batch|models)/?$', frontend, name='frontend-route'),
]

# Serve media files in development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
