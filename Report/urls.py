from django.conf.urls.static import static
from django.contrib import admin
from django.conf.urls import url


from files.views import ProfileImageView, ResultsView

urlpatterns = [

    url(r'^$', ProfileImageView.as_view(), name='profile_image_upload'),
    url(r'^check/$', ProfileImageView.as_view(), name='profile_image_upload1'),
    url(r'^results/$', ResultsView, name='results'),
# url(r'^admin/', include(admin.site.urls)),
# ) + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT),

    # path('admin/', admin.site.urls),
]