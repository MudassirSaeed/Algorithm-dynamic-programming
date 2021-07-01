from django.urls import path
from algos import views

app_name = "algos"

urlpatterns =[
    path('',views.get_main_page,name="get_main_page"),
    path('lcs',views.lcs,name="lcs"),
    path('mcm',views.mcm,name="mcm"),
    path('word_break',views.findWord,name="word_break"),
    path('knapsack',views.knapsack,name="knapsack"),
    path('coin_change',views.coin_change,name="coin_change"),
    path('lis',views.lis,name="lis"),
    path('check_partition',views.check_partition,name="check_partition"),
    path('rod_cutting',views.rod_cutting,name="rod_cutting"),
    path('scs',views.scs,name="scs"),
    path('edit_distance',views.edit_distance,name="edit_distance")
]