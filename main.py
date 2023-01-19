import math

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
B = 0.035  # [m^3/s]
h_min = 0.0  # [m]
h_max = 5.0  # [m]
u_min = 0.0  # minimalny przepływ
u_max = 10.0  # maksymalny przepływ
kp = 0.015  # wzmocnienie
Ti = 5  # [s]
N = int(T/Tp) + 1


# TABLES
U = [0.0]    # zmienne do zaworu regulacyjnego
e = [0.0]
h = [0.0]
q_in = [0.08]
q_out = [0.0]
t = [0.0]
e_1 = [0.0]
e_2 = [0.0]
u_pi_1 = [0.025]
u_pi_2 = [0.025]
V = [3.0]
V_max = A * math.pi * h_max
V_min = 0.0
c_1 = [0.91]
c_2 = [0.12]
c = [0.5]
q_in_1 = [0.025]
q_in_2 = [0.025]


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
    c_1 = [0.91]
    q_in_1 = [0.025]
    q_in_2 = [0.025]
    c_2 = [0.12]
    c = [0.5]
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

def p_regulator(hsp):
    global U, e, h, q_in, q_out, t
    refresh_tables()
    print("do")
    for n in range(1, N):
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
    for n in range(1, N):

        t.append(n * Tp)
        e.append(csp-c[n-1])
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
    return ("nothing")\

@app.route('/p_regulator_execute',methods=['GET'])
def p_regulator_execute():
    hsp = request.args.get('hsp')
    p_regulator(float(hsp))
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
    print(e_plot.head(100))
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
    print(e_plot.head(100))
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
    print(e_plot.head(100))
    e_sns=sns.lineplot(x = "Czas", y = "Wartość uchybu", data=e_plot.sample(1000), ax=ax[3,0], color='g')
    fig.tight_layout()

    canvas = FigureCanvas(fig)
    img = io.BytesIO()
    fig.savefig("static/plot.png")
    # img.seek(0)

    return "success"


if __name__ == "__main__":
    app.run(debug=True)
