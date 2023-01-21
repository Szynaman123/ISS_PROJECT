import math
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from flask import Flask, send_file, render_template
from flask_bootstrap import Bootstrap
from flask_nav import Nav
from flask_nav.elements import *
from dominate.tags import img
import pandas as pd
import io
import numpy as np
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt

topbar = Navbar(View('Home', 'get_home'),
                View('Charts', 'get_charts'))

nav = Nav()
nav.register_element('top', topbar)

# CONSTANTS
T = 3600  # [s]
Tp = 0.1  # [s]
A = 1.5  # [m]
B = 0.055  # [m^3/s]
h_min = 0.0  # [m]
h_max = 15.0  # [m]
u_min = 0.0  # minimalny przepływ
u_max = 10.0  # maksymalny przepływ
q_in_max = 0.1  # maksymalny przepływ
kp = 0.015  # wzmocnienie
Ti = 5  # [s]
N = int(T/Tp) + 1


# TABLES
U = [0.0]    # zmienne do zaworu regulacyjnego
e = [0.0]
ce = [0.0]
h = [0.0]
q_in = [0.08]
q_out = [0.0]
t = [0.0]
e_1 = [0.0]
e_2 = [0.0]
u_pi_1 = [0.0]
u_pi_2 = [0.025]
V = [3.0]
V_max = A * math.pi * h_max
V_min = 0.0
c_1 = [0.9]
c_2 = [0.1]
c = [0.0]
q_in_1 = [0.0]
q_in_2 = [0.025]
przedzialy_uchyb=[[-1.0,-1.0,-0.5],[-1.0,-0.5,0],[-0.5,0,0.5],[0,0.5,1.0],[0.5,1.0,1.0]]
przedzialy_c_uchyb=[[-2.0,-2.0,-1.0],[-2.0,-1.0,0],[-1.0,0,1.0],[0,1.0,2.0],[1.0,2.0,2.0]]
przedzialy=[[0,0,0.0125],[0,0.0125,0.025],[0.0125,0.025,0.0375],[0.025,0.0375,0.05],[0.0375,0.05,0.05]]
mu_e=[]
mu_ce=[]
mu_przedzialy=[]
table=[ [0, 0, 0, 1, 2],
        [0, 0, 1, 2, 3],
        [0,	1, 2, 3, 4],
        [1,	2, 3, 4, 4],
        [2,	3, 4, 4, 4]]


def fuzzy(e,ce):
    global mu_ce,mu_e,mu_przedzialy
    mu_e = []
    mu_ce = []
    mu_przedzialy = []
    for i in range(len(przedzialy_uchyb)):
        if e >= przedzialy_uchyb[i][0] and e <= przedzialy_uchyb[i][2]:
            if i==0:
                x=(przedzialy_uchyb[i][2]-e)/(przedzialy_uchyb[i][2]-przedzialy_uchyb[i][1])
            else:
                x=max(min((e-przedzialy_uchyb[i][0])/(przedzialy_uchyb[i][1]-przedzialy_uchyb[i][0]),
                      (przedzialy_uchyb[i][2]-e)/(przedzialy_uchyb[i][2]-przedzialy_uchyb[i][1])),0)
            if i+1==4:
                y=(przedzialy_uchyb[i][2]-e)/(przedzialy_uchyb[i][2]-przedzialy_uchyb[i][1])
            else:
                y=max(min((e-przedzialy_uchyb[i+1][0])/(przedzialy_uchyb[i+1][1]-przedzialy_uchyb[i+1][0]),
                      (przedzialy_uchyb[i+1][2]-e)/(przedzialy_uchyb[i+1][2]-przedzialy_uchyb[i+1][1])),0)
            # x=(przedzialy_uchyb[i][2]-e)/(przedzialy_uchyb[i][2]-przedzialy_uchyb[i][1])
            # y=(e-przedzialy_uchyb[i+1][0])/(przedzialy_uchyb[i+1][1]-przedzialy_uchyb[i+1][0])
            mu_e.append({"value":x,"index":i})
            mu_e.append({"value":y,"index":i+1})
            break
    for i in range(len(przedzialy_c_uchyb)):
        if ce >= przedzialy_c_uchyb[i][0] and ce <= przedzialy_c_uchyb[i][2]:
            if i==0:
                x=(przedzialy_c_uchyb[i][2]-ce)/(przedzialy_c_uchyb[i][2]-przedzialy_c_uchyb[i][1])
            else:
                x=max(min((ce-przedzialy_c_uchyb[i][0])/(przedzialy_c_uchyb[i][1]-przedzialy_c_uchyb[i][0]),
                      (przedzialy_c_uchyb[i][2]-ce)/(przedzialy_c_uchyb[i][2]-przedzialy_c_uchyb[i][1])),0)
            if i+1==4:
                y=(przedzialy_c_uchyb[i][2]-ce)/(przedzialy_c_uchyb[i][2]-przedzialy_c_uchyb[i][1])
            else:
                y=max(min((ce-przedzialy_c_uchyb[i+1][0])/(przedzialy_c_uchyb[i+1][1]-przedzialy_c_uchyb[i+1][0]),
                      (przedzialy_c_uchyb[i+1][2]-ce)/(przedzialy_c_uchyb[i+1][2]-przedzialy_c_uchyb[i+1][1])),0)
            mu_ce.append({"value":x,"index":i})
            mu_ce.append({"value":y,"index":i+1})
            break

    r_1 = min(mu_e[0]["value"],mu_ce[0]["value"])
    r_2 = min(mu_e[1]["value"],mu_ce[0]["value"])
    r_3 = min(mu_e[0]["value"],mu_ce[1]["value"])
    r_4 = min(mu_e[1]["value"],mu_ce[1]["value"])


    mu_przedzialy.append({"value":r_1,"index":table[mu_e[0]["index"]][mu_ce[0]["index"]]})
    mu_przedzialy.append({"value":r_2,"index":table[mu_e[1]["index"]][mu_ce[0]["index"]]})
    mu_przedzialy.append({"value":r_3,"index":table[mu_e[0]["index"]][mu_ce[1]["index"]]})
    mu_przedzialy.append({"value":r_4,"index":table[mu_e[1]["index"]][mu_ce[1]["index"]]})

    for i in range(len(mu_przedzialy)-1):
        if mu_przedzialy[i]["index"]==mu_przedzialy[i+1]["index"]:
            var=max(mu_przedzialy[i]["value"],mu_przedzialy[i+1]["value"])
            mu_przedzialy[i]["value"]=var
            mu_przedzialy[i+1]["value"]=var
    res_list = [i for n, i in enumerate(mu_przedzialy)
                if i not in mu_przedzialy[n + 1:]]

    areas=[]
    areas_middle_multiply=[]
    for i in range(len(res_list)):
        if res_list[i]["index"]==0:
            g= abs(przedzialy[res_list[i]["index"]][2])-res_list[i]["value"]*abs(przedzialy[res_list[i]["index"]][2]-przedzialy[res_list[i]["index"]][1])
            area=(res_list[i]["value"]/2)*(przedzialy[res_list[i]["index"]][2]-g)
        elif res_list[i]["index"]==4:
            g= przedzialy[res_list[i]["index"]][1]-res_list[i]["value"]*(przedzialy[res_list[i]["index"]][1]-przedzialy[res_list[i]["index"]][0])
            area=(res_list[i]["value"]/2)*(przedzialy[res_list[i]["index"]][1]-g)
        else:
            g1=res_list[i]["value"]*abs(przedzialy[res_list[i]["index"]][1]-przedzialy[res_list[i]["index"]][0])+abs(przedzialy[res_list[i]["index"]][0])
            g2=abs(przedzialy[res_list[i]["index"]][2])-(res_list[i]["value"]*abs(przedzialy[res_list[i]["index"]][2]-przedzialy[res_list[i]["index"]][1]))
            g=g2-g1
        area=(res_list[i]["value"]/2)*(abs(przedzialy[res_list[i]["index"]][2]-przedzialy[res_list[i]["index"]][0])+g)
        areas.append(area)
        areas_middle_multiply.append(area*przedzialy[res_list[i]["index"]][1])
    u=sum(areas_middle_multiply)/sum(areas)
    # for  i in range(len(res_list)):
    #     if res_list[i]["value"]>max_u["value"]:
    #         max_u=res_list[i]
    #
    # y_max=(przedzialy[max_u["index"]][2]-przedzialy[max_u["index"]][1])*(1-max_u["value"])+przedzialy[max_u["index"]][1]
    # y_min=max(przedzialy[max_u["index"]][1]-((przedzialy[max_u["index"]][1]-przedzialy[max_u["index"]][0])*(1-max_u["value"])),0)
    # if max_u["index"]==4:
    #     u=y_min
    # else:
    #     u=(y_max+y_min)/2
    return min(u,q_in_max)



# e_pred = ctrl.Antecedent(np.arange(-1, 1, 0.0001), 'e')
# ce_pred = ctrl.Antecedent(np.arange(-2, 2, 0.0001), 'ce')
# u_in_pred = ctrl.Consequent(np.arange(0, 5.01, 0.0001), 'U')
# e_pred.automf(3)
# ce_pred.automf(3)
# u_in_pred.automf(3)
# # u_in_pred['poor'] = fuzz.trimf(u_in_pred.universe, [0, 0, 1.5])
# # u_in_pred['mediocre'] = fuzz.trimf(u_in_pred.universe, [0, 1.5, 2.5])
# # u_in_pred['average'] = fuzz.trimf(u_in_pred.universe, [1.5, 2.5, 3.5])
# # u_in_pred['decent'] = fuzz.trimf(u_in_pred.universe, [2.5, 3.5, 5])
# # u_in_pred['good'] = fuzz.trimf(u_in_pred.universe, [3.5, 5 , 5 ])
# rule1 = ctrl.Rule(e_pred['poor'] | ce_pred['poor'], u_in_pred['poor'])
# # rule2 = ctrl.Rule(e_pred['poor'] | ce_pred['mediocre'], u_in_pred['poor'])
# rule3 = ctrl.Rule(e_pred['poor'] | ce_pred['average'], u_in_pred['poor'])
# # rule4 = ctrl.Rule(e_pred['poor'] | ce_pred['decent'], u_in_pred['mediocre'])
# rule5 = ctrl.Rule(e_pred['poor'] | ce_pred['good'], u_in_pred['average'])
#
# # rule6 = ctrl.Rule(e_pred['mediocre'] | ce_pred['poor'], u_in_pred['poor'])
# # rule7 = ctrl.Rule(e_pred['mediocre'] | ce_pred['mediocre'], u_in_pred['poor'])
# # rule8 = ctrl.Rule(e_pred['mediocre'] | ce_pred['average'], u_in_pred['mediocre'])
# # rule9 = ctrl.Rule(e_pred['mediocre'] | ce_pred['decent'], u_in_pred['average'])
# # rule10 = ctrl.Rule(e_pred['mediocre'] | ce_pred['good'], u_in_pred['decent'])
#
# rule11 = ctrl.Rule(e_pred['average'] | ce_pred['poor'], u_in_pred['poor'])
# # rule12 = ctrl.Rule(e_pred['average'] | ce_pred['mediocre'], u_in_pred['mediocre'])
# rule13 = ctrl.Rule(e_pred['average'] | ce_pred['average'], u_in_pred['average'])
# # rule14 = ctrl.Rule(e_pred['average'] | ce_pred['decent'], u_in_pred['decent'])
# rule15 = ctrl.Rule(e_pred['average'] | ce_pred['good'], u_in_pred['good'])
#
# # rule16 = ctrl.Rule(e_pred['decent'] | ce_pred['poor'], u_in_pred['mediocre'])
# # rule17 = ctrl.Rule(e_pred['decent'] | ce_pred['mediocre'], u_in_pred['average'])
# # rule18 = ctrl.Rule(e_pred['decent'] | ce_pred['average'], u_in_pred['decent'])
# # rule19 = ctrl.Rule(e_pred['decent'] | ce_pred['decent'], u_in_pred['good'])
# # rule20 = ctrl.Rule(e_pred['decent'] | ce_pred['good'], u_in_pred['good'])
#
# rule21 = ctrl.Rule(e_pred['good'] | ce_pred['poor'], u_in_pred['average'])
# # rule22 = ctrl.Rule(e_pred['good'] | ce_pred['mediocre'], u_in_pred['decent'])
# rule23 = ctrl.Rule(e_pred['good'] | ce_pred['average'], u_in_pred['good'])
# # rule24 = ctrl.Rule(e_pred['good'] | ce_pred['decent'], u_in_pred['good'])
# rule25 = ctrl.Rule(e_pred['good'] | ce_pred['good'], u_in_pred['good'])
# u_in_ctrl = ctrl.ControlSystem([rule1, rule3,rule5,rule11,
#                                 rule13,  rule15,rule21,  rule23,rule25])
# # u_in_ctrl = ctrl.ControlSystem([rule1, rule2, rule3,rule4,rule5, rule6, rule7,rule8,rule9, rule10, rule11,rule12,
# #                                 rule13, rule14, rule15,rule16,rule17, rule18, rule19,rule20,rule21, rule22, rule23,rule24,rule25])
# u_in = ctrl.ControlSystemSimulation(u_in_ctrl)
#
# u_in.input['e'] = 0.99
# u_in.input['ce'] =1.99
#
# u_in.compute()
# print(u_in.output['U'])
# u_in_pred.view(sim=u_in)

# quality = ctrl.Antecedent(np.arange(-1.0, 1.0, 0.5), 'quality')
# service = ctrl.Antecedent(np.arange(-2.0, 2.0, 0.5), 'service')
# tip = ctrl.Consequent(np.arange(0.0, 5.5, 0.5), 'tip')

# Auto-membership function population is possible with .automf(3, 5, or 7)
# quality.automf(3)
# service.automf(3)

# Custom membership functions can be built interactively with a familiar,
# Pythonic API
# tip['poor'] = fuzz.trimf(tip.universe, [0, 0, 2.5])
# tip['average'] = fuzz.trimf(tip.universe, [0, 2.5, 5])
# tip['good'] = fuzz.trimf(tip.universe, [2.5, 5, 5])
# rule1 = ctrl.Rule(quality['poor'] | service['poor'], tip['poor'])
# rule2 = ctrl.Rule(service['average'], tip['average'])
# rule3 = ctrl.Rule(service['good'] | quality['good'], tip['good'])
# tipping_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
# tipping = ctrl.ControlSystemSimulation(tipping_ctrl)
# tipping.input['quality'] = 0.3
# tipping.input['service'] = 0.1

# Crunch the numbers
# tipping.compute()
# print (tipping.output['tip'])
# tip.view(sim=tipping)
# plt.show()

def refresh_tables():
    global U,e,h,q_in,q_out,t,q_in_1, q_in_2, u_pi_1, u_pi_2, e_1, e_2, V, c_1, c_2, c
    U = [0.0]
    e_1 = [0.0]
    e_2 = [0.0]
    u_pi_1 = [0.025]
    u_pi_2 = [0.025]
    V = [3.0]
    e = [0.0]
    h = [0.0]
    c_1 = [0.92]
    q_in_1 = [0.0]
    q_in_2 = [0.025]
    c_2 = [0.17]
    c = [0.0]
    q_in = [0.08]
    q_out = [0.0]
    t = [0.0]
def no_regulator(hsp):
    global U, e, h, q_in, q_out, t
    refresh_tables()
    print("do")
    for n in range(1, N):
        t.append(n * Tp)
        q_in.append(q_in[n - 1])
        h.append(min(max(Tp * (q_in[n - 1] - q_out[n - 1]) / A + h[n-1], h_min), h_max))
        e.append(float(hsp) - float(h[n - 1]))
        q_out.append(B * sqrt(h[n - 1]))
    print("finish")

def reg_pi(reg_list, e_list):
    global U, e, h, q_in, q_out, t
    reg_list.append(max(min(kp * (e_list[-1] + Tp * sum(e_list) / Ti), u_max), u_min))
    return reg_list

def p_regulator(hsp,dist,dist_time):
    global U, e, h, q_in, q_out, t
    dist_in = True
    if dist < 0.0:
        dist_in= False
    refresh_tables()
    print(dist_time)
    for n in range(1, N):
        if n >= (dist_time*10) and dist != 0.0:
            t.append(n * Tp)
            e.append(hsp - h[n - 1])
            U = reg_pi(U, e)
            if dist_in:
                U[-1] += dist
            q_in.append(q_in[n - 1])

            h.append(min(max(Tp * (U[n - 1] - q_out[n - 1]) / A + h[n - 1], h_min), h_max))
            if not dist_in:
                q_out.append((B * sqrt(h[n - 1]))+abs(dist))
            else:
                q_out.append(B * sqrt(h[n - 1]))
        else:
            t.append(n * Tp)
            e.append(hsp-h[n-1])
            U=reg_pi(U,e)
            q_in.append(q_in[n - 1])
            h.append(min(max(Tp * (U[n-1] - q_out[n - 1]) / A + h[n-1], h_min), h_max))

            q_out.append(B * sqrt(h[n - 1]))
    print("finish")





def mixing(hsp):
    global U, e, h, q_in, q_out, t, q_in_1, q_in_2, u_pi_1, u_pi_2
    refresh_tables()
    csp=hsp
    print("do")
    for n in range(1, N):
        t.append(n * Tp)
        e[0]=csp-c[n-1]
        e.append(csp-c[n-1])
        wspolczynnik=((csp*0.025)-(0.025*c_2[-1]))/(c_1[-1]-csp)
        ce.append(e[n]-e[n-1])
        # u_pi_1.append(fuzzy(e[n],ce[n]))
        # u_pi_2 = reg_pi(u_pi_2, e_2)
        V.append(min(max((q_in_1[n - 1] + q_in_2[n - 1] - q_out[n - 1]) * Tp + V[n - 1], V_min), V_max))
        q_in_1.append(fuzzy(e[n],ce[n])*wspolczynnik/0.025)
        q_in_2.append(q_in_2[n - 1])
        h.append(min(max(Tp * (q_in_1[n - 1] + q_in_2[n - 1] - q_out[n - 1]) / A + h[n - 1], h_min), h_max))
        c.append((1.0 / V[n - 1]) * (q_in_1[n - 1] * (c_1[n - 1] - c[n - 1]) + (q_in_2[n - 1] * (c_2[n - 1] - c[n - 1]))) * Tp + c[n - 1])
        c_1.append(c_1[n-1])
        c_2.append(c_2[n-1])
        q_out.append(B * sqrt(h[n - 1]))
        print((e[n],ce[n],q_in_1[n]))
    print("finish")
    print(c[-1])
def reg_mixing(hsp):
    global U, e, ce, h, q_in, q_out, t, q_in_1, q_in_2, u_pi_1, u_pi_2
    refresh_tables()
    csp=hsp
    for n in range(1, N):

        t.append(n * Tp)
        e.append(csp-c[n-1])
        ce.append(e[n]-e[n-1])
        ce[n]=0.01
        u_pi_1 = reg_pi(u_pi_1, e_1)
        u_pi_2 = reg_pi(u_pi_2, e_2)
        V.append(min(max((q_in_1[n - 1] + q_in_2[n - 1] - q_out[n - 1]) * Tp + V[n - 1], V_min), V_max))
        q_in_1.append(q_in_1[n - 1])
        q_in_2.append(q_in_2[n - 1])
        h.append(min(max(Tp * (q_in_1[n - 1] + q_in_2[n - 1] - q_out[n - 1]) / A + h[n - 1], h_min), h_max))
        c.append((1.0 / V[n - 1]) * (q_in_1[n - 1] * (c_1[n - 1] - c[n - 1]) + (q_in_2[n - 1] * (c_2[n - 1] - c[n - 1]))) * Tp + c[n - 1])
        c_1.append(c_1[n-1])
        c_2.append(c_2[n-1])
        q_out.append(B * sqrt(h[n - 1]))

app = Flask(__name__)

@app.route('/no_regulator_execute',methods=['GET'])
def no_regulator_execute():
    hsp = request.args.get('hsp')
    no_regulator(float(hsp))
    return ("nothing")

@app.route('/p_regulator_execute',methods=['GET'])
def p_regulator_execute():
    hsp = request.args.get('hsp')
    dist = request.args.get('dist')
    dist_time = request.args.get('dist_time')
    p_regulator(float(hsp),float(dist),int(dist_time))
    return ("nothing")

@app.route('/mixing_execute',methods=['GET'])
def mixing_execute():
    hsp = request.args.get('hsp')
    mixing(float(hsp))
    return ("nothing")


@app.route('/home', methods=['GET'])
def get_home():
    return (render_template('home.html'))

@app.route('/', methods=['GET'])
@app.route('/labs', methods=['GET'])
def get_labs():
    return (render_template('labs.html'))

@app.route('/no_reg', methods=['GET'])
def get_no_reg():
    return (render_template('no_regulator.html'))

@app.route('/mixing', methods=['GET'])
def get_mixing():
    return (render_template('mixing.html'))



@app.route('/p_reg', methods=['GET'])
def get_p_reg():
    return (render_template('p_regulator.html'))


nav.init_app(app)

@app.route('/visualize')
def visualize():
    global U, e, h, q_in, q_out, t
    # no_regulator()
    sns.set_theme(style="darkgrid")
    h_plot = pd.DataFrame({"Wysokość cieczy w zbiorniku": h, "Czas": t})
    i_plot = pd.DataFrame({"Wartość dopływu": q_in, "Czas": t})
    o_plot = pd.DataFrame({"Natężenie odpływu": q_out, "Czas": t})
    e_plot = pd.DataFrame({"Wartość uchybu": e, "Czas": t})
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12,5))
    h_sns=sns.lineplot(x = "Czas", y = "Wysokość cieczy w zbiorniku", data=h_plot.sample(1000) , ax=ax[0,0], color='r')
    h_sns.set_yticks(range(0,6))
    sns.lineplot(x = "Czas", y = "Wartość dopływu", data=i_plot.sample(1000), ax=ax[0,1] , color='y')
    o_sns=sns.lineplot(x = "Czas", y = "Natężenie odpływu", data=o_plot.sample(1000),  ax=ax[1,0], color='b')
    o_sns.set_yticks(np.arange(0.0,0.12,0.02))
    e_sns=sns.lineplot(x = "Czas", y = "Wartość uchybu", data=e_plot.sample(1000), ax=ax[1,1], color='g')
    e_sns.set_yticks(range(-5,6))
    fig.tight_layout()

    canvas = FigureCanvas(fig)
    img = io.BytesIO()
    fig.savefig("static/plot.png")
    # img.seek(0)

    return "success"


@app.route('/visualize_P')
def visualize_P():
    global U, e, h, q_in, q_out, t
    # no_regulator()
    sns.set_theme(style="darkgrid")
    h_plot = pd.DataFrame({"Wysokość cieczy w zbiorniku": h, "Czas": t})
    i_plot = pd.DataFrame({"Wartość dopływu": U, "Czas": t})
    o_plot = pd.DataFrame({"Natężenie odpływu": q_out, "Czas": t})
    e_plot = pd.DataFrame({"Wartość uchybu": e, "Czas": t})
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12,5))
    h_sns=sns.lineplot(x = "Czas", y = "Wysokość cieczy w zbiorniku", data=h_plot.sample(1000) , ax=ax[0,0], color='r')
    h_sns.set_yticks(range(0,6))
    sns.lineplot(x = "Czas", y = "Wartość dopływu", data=i_plot.sample(1000), ax=ax[0,1] , color='y')
    o_sns=sns.lineplot(x = "Czas", y = "Natężenie odpływu", data=o_plot.sample(1000),  ax=ax[1,0], color='b')
    o_sns.set_yticks(np.arange(0.0,0.12,0.02))
    e_sns=sns.lineplot(x = "Czas", y = "Wartość uchybu", data=e_plot.sample(1000), ax=ax[1,1], color='g')
    e_sns.set_yticks(range(-5,6))
    fig.tight_layout()

    canvas = FigureCanvas(fig)
    img = io.BytesIO()
    fig.savefig("static/plot.png")
    # img.seek(0)

    return "success"

@app.route('/visualize_M')
def visualize_M():
    global U, e, h, q_in, q_out, t
    # no_regulator()
    sns.set_theme(style="darkgrid")
    h_plot = pd.DataFrame({"Wysokość cieczy w zbiorniku": h, "Czas": t})
    i_plot_1 = pd.DataFrame({"Wartość dopływu 1": q_in_1, "Czas": t})
    i_plot_2 = pd.DataFrame({"Wartość dopływu 2": q_in_2, "Czas": t})
    o_plot = pd.DataFrame({"Natężenie odpływu": q_out, "Czas": t})
    e_plot = pd.DataFrame({"Wartość uchybu": e, "Czas": t})
    c_plot = pd.DataFrame({"Stężenie składnika": c, "Czas": t})
    v_plot = pd.DataFrame({"Objętość": V, "Czas": t})
    fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(12,8))
    h_sns=sns.lineplot(x = "Czas", y = "Wysokość cieczy w zbiorniku", data=h_plot.sample(1000) , ax=ax[0,0], color='r')
    h_sns.set_yticks(range(0,6))
    sns.lineplot(x = "Czas", y = "Wartość dopływu 1", data=i_plot_1.sample(1000), ax=ax[0,1] , color='y')
    sns.lineplot(x = "Czas", y = "Wartość dopływu 2", data=i_plot_2.sample(1000), ax=ax[1,0] , color='y')
    sns.lineplot(x = "Czas", y = "Stężenie składnika", data=c_plot.sample(1000), ax=ax[1,1] , color='y')
    sns.lineplot(x = "Czas", y = "Objętość", data=v_plot.sample(1000), ax=ax[2,0] , color='y')
    o_sns=sns.lineplot(x = "Czas", y = "Natężenie odpływu", data=o_plot.sample(1000),  ax=ax[2,1], color='b')
    o_sns.set_yticks(np.arange(0.0,0.12,0.02))
    e_sns=sns.lineplot(x = "Czas", y = "Wartość uchybu", data=e_plot.sample(1000), ax=ax[3,0], color='g')
    fig.tight_layout()

    canvas = FigureCanvas(fig)
    img = io.BytesIO()
    fig.savefig("static/plot.png")
    # img.seek(0)

    return "success"


if __name__ == "__main__":
    print(fuzzy(-0.2323863964588086,0.0))
    app.run(debug=False)
