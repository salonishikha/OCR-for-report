from django.conf.urls.static import static
from django.contrib import admin
from django.conf.urls import url
from django.urls import include

from files.views import ProfileImageView

urlpatterns = [

    url(r'^$', ProfileImageView.as_view(), name='profile_image_upload'),
# url(r'^admin/', include(admin.site.urls)),
# ) + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT),

    # path('admin/', admin.site.urls),
]