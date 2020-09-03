# -*- coding:utf-8 -*-
'''
	@author: Yueyang Li
	@date:2020-09-03
'''
import numpy as np
import pandas as pd
from scipy.optimize import fmin_slsqp
from tkinter import *
from tkinter.filedialog import askdirectory
from tkinter import messagebox


class DEA(object):

    def __init__(self, inputs, outputs):

        # supplied data
        self.inputs = inputs
        self.outputs = outputs

        # parameters
        self.n = inputs.shape[0]
        self.m = inputs.shape[1]
        self.r = outputs.shape[1]

        # iterators
        self.unit_ = range(self.n)
        self.input_ = range(self.m)
        self.output_ = range(self.r)

        # result arrays
        self.output_w = np.zeros((self.r, 1), dtype=np.float)  # output weights
        self.input_w = np.zeros((self.m, 1), dtype=np.float)  # input weights
        self.lambdas = np.zeros((self.n, 1), dtype=np.float)  # unit efficiencies
        self.efficiency = np.zeros_like(self.lambdas)  # thetas
    def __efficiency(self, unit):
        # compute efficiency
        denominator = np.dot(self.inputs, self.input_w)
        numerator = np.dot(self.outputs, self.output_w)

        return (numerator/denominator)[unit]
    def __target(self, x, unit):

        in_w, out_w, lambdas = x[:self.m], x[self.m:(self.m+self.r)], x[(self.m+self.r):]  # unroll the weights
        denominator = np.dot(self.inputs[unit], in_w)
        numerator = np.dot(self.outputs[unit], out_w)

        return numerator/denominator

    def __constraints(self, x, unit):

        in_w, out_w, lambdas = x[:self.m], x[self.m:(self.m+self.r)], x[(self.m+self.r):]  # unroll the weights
        constr = []  # init the constraint array

        # for each input, lambdas with inputs
        for input in self.input_:
            t = self.__target(x, unit)
            lhs = np.dot(self.inputs[:, input], lambdas)
            cons = t*self.inputs[unit, input] - lhs
            constr.append(cons)

        # for each output, lambdas with outputs
        for output in self.output_:
            lhs = np.dot(self.outputs[:, output], lambdas)
            cons = lhs - self.outputs[unit, output]
            constr.append(cons)

        # for each unit
        for u in self.unit_:
            constr.append(lambdas[u])

        return np.array(constr)

    def __optimize(self):

        d0 = self.m + self.r + self.n
        # iterate over units
        for unit in self.unit_:
            # weights
            x0 = np.random.rand(d0) - 0.5
            x0 = fmin_slsqp(self.__target, x0, f_ieqcons=self.__constraints, args=(unit,))
            # unroll weights
            self.input_w, self.output_w, self.lambdas = x0[:self.m], x0[self.m:(self.m+self.r)], x0[(self.m+self.r):]
            self.efficiency[unit] = self.__efficiency(unit)

    def fit(self):
        self.__optimize()  # optimize
        return self.efficiency


class BCC_DEA(object):

    def __init__(self, inputs, outputs, eps):

        # supplied data
        self.inputs = inputs
        self.outputs = outputs

        # parameters
        self.n = inputs.shape[0]
        self.m = inputs.shape[1]
        self.r = outputs.shape[1]
        self.eps = eps
        # iterators
        self.unit_ = range(self.n)
        self.input_ = range(self.m)
        self.output_ = range(self.r)

        # result arrays

        self.S_o = np.zeros((self.r, 1), dtype=np.float)  # output weights
        self.S_m = np.zeros((self.m, 1), dtype=np.float)  # input weights
        self.lambdas = np.zeros((self.n, 1), dtype=np.float)  # unit efficiencies
        self.efficiency = np.zeros_like(self.lambdas)  # thetas


    def __target(self, x, unit):

        # in_w, out_w, lambdas = x[:self.m],   # unroll the weights
        # denominator = np.dot(self.inputs[unit], in_w)
        # numerator = np.dot(self.outputs[unit], out_w)
        theta, S_m, S_o, lambdas = x[0], x[1: (1 + self.m)], x[(1 + self.m):(1 + self.m+self.r)], x[(1 + self.m+self.r):]

        return theta - self.eps * (np.dot(np.ones(self.m), S_m) + np.dot(np.ones(self.r), S_o))


    def __equalconstriants(self, x, unit):
        theta, S_m, S_o, lambdas = x[0], x[1: (1 + self.m)], x[(1 + self.m):(1 + self.m+self.r)], x[(1 + self.m+self.r):]
        constr = []

        for input in self.input_:
            theta = self.__target(x, unit)
            lhs = np.dot(self.inputs[:, input], lambdas)
            cons = theta * self.inputs[unit, input] - lhs - S_m[input]
            constr.append(cons)

        # for each output, lambdas with outputs
        for output in self.output_:
            lhs = np.dot(self.outputs[:, output], lambdas)
            cons = lhs - S_o[output] - self.outputs[unit, output]
            constr.append(cons)

        constr.append(np.sum(lambdas) - 1)

        return np.array(constr)


    def __constraints(self, x, unit):

        return x[1:]

    def __optimize(self):

        d0 = 1 + self.m + self.r + self.n
        # iterate over units
        for unit in self.unit_:
            # weights
            x0 = np.random.rand(d0) - 0.5
            x0 = fmin_slsqp(self.__target, x0, f_eqcons=self.__equalconstriants, f_ieqcons=self.__constraints, args=(unit,))
            # unroll weights
            self.theta, self.S_m, self.S_o, self.lambdas = x0[0], x0[1: (1 + self.m)], x0[(1 + self.m):(1 + self.m+self.r)], x0[(1 + self.m+self.r):]
            self.efficiency[unit] = self.theta

    def fit(self):
        self.__optimize()  # optimize
        return self.efficiency

class SBM_DEA(object):

    def __init__(self, inputs, outputs):

        # supplied data
        self.inputs = inputs
        self.outputs = outputs

        # parameters
        self.n = inputs.shape[0]# 观测数
        self.m = inputs.shape[1]# x的个数
        self.r = outputs.shape[1]# y的个数

        # iterators
        self.unit_ = range(self.n)# 观测数
        self.input_ = range(self.m)# x 的个数
        self.output_ = range(self.r)# y的个数

        # result arrays
        self.S_o = np.zeros((self.r, 1), dtype=np.float)  # output weights
        self.S_i = np.zeros((self.m, 1), dtype=np.float)  # input weights
        self.lambdas = np.zeros((self.n, 1), dtype=np.float)  # unit efficiencies
        self.efficiency = np.zeros_like(self.lambdas)  # thetas

    def __efficiency(self, unit):
        # compute efficiency
        numerator = np.dot(1 / self.inputs, self.S_i) / self.m
        denominator = np.dot(1 / self.outputs, self.S_o) / self.r

        return ((1 - numerator) / (1 + denominator))[unit]
    def __target(self, x, unit):
        '''
        此处的目标函数为 W_X^T X / W_Y^T Y
        '''
        S_i, S_o, lambdas = x[:self.m], x[self.m:(self.m+self.r)], x[(self.m+self.r):]  # unroll the weights
        numerator = np.dot(1 / self.inputs[unit], S_i) / self.m
        denominator = np.dot(1 / self.outputs[unit], S_o) / self.r

        return (1 - numerator) / (1 + denominator)

    def __equalconstriants(self, x, unit):
        S_i, S_o, lambdas = x[:self.m], x[self.m:(self.m + self.r)], x[(self.m + self.r):]  # unroll the weights        constr = []
        constr = []

        for input in self.input_:
            lhs = np.dot(self.inputs[:, input], lambdas)
            cons = self.inputs[unit, input] - lhs - S_i[input]
            constr.append(cons)

        # for each output, lambdas with outputs
        for output in self.output_:
            lhs = np.dot(self.outputs[:, output], lambdas)
            cons = lhs - S_o[output] - self.outputs[unit, output]
            constr.append(cons)

        return np.array(constr)

    def __constraints(self, x, unit):
        return np.array(x)

    def __optimize(self):

        d0 = self.m + self.r + self.n
        # iterate over units
        for unit in self.unit_:
            # weights
            x0 = np.random.rand(d0) - 0.5
            x0 = fmin_slsqp(self.__target, x0, f_eqcons=self.__equalconstriants, f_ieqcons=self.__constraints, args=(unit,))# 这里限定条件大于等于0
            # unroll weights
            self.S_i, self.S_o, self.lambdas = x0[:self.m], x0[self.m:(self.m+self.r)], x0[(self.m+self.r):]
            self.efficiency[unit] = self.__efficiency(unit)

    def fit(self):
        self.__optimize()  # optimize
        return self.efficiency



#%% UI 界面化
# 读取文件路径
## BCC模型需要eps, S+, S-三个超参数


root = Tk()
root.title('DEA method caculation for Li Duo')
root.geometry('500x600')

var = StringVar()
l = Label(root, bg='yellow', width=20, text='empty')
l.pack()


def print_selection():
    l.config(text='you have selected ' + var.get())
le1 = Label(root,text="自变量路径(csv文件）:")
le2 = Label(root,text="因变量路径(csv文件）:")
le3 = Label(root,text="eps(BCC-DEA模型采用), 若无填写0")
le4 = Label(root,text="结果存储路径")
e1 = Entry(root, show=None, font=('Arial', 14))   
e2 = Entry(root, show=None, font=('Arial', 14))  # 显示成明文形式
e3 = Entry(root, show=None, font=('Arial', 14))  # 显示成明文形式
e4 = Entry(root, show=None, font=('Arial', 14))  # 显示成明文形式
le1.pack()
e1.pack()
le2.pack()
e2.pack()
le3.pack()
e3.pack()
le4.pack()
e4.pack()

r1 = Radiobutton(root, text='Orignal DEA', variable=var, value='DEA', command=print_selection)
r1.pack()
r2 = Radiobutton(root, text='BCC-DEA', variable=var, value='BCC_DEA', command=print_selection)
r2.pack()
r3 = Radiobutton(root, text='SBM-DEA', variable=var, value='SBM_DEA', command=print_selection)
r3.pack()
def runDEA():
    xpath = e1.get()
    ypath = e2.get()
    eps = float(e3.get())
    savepath = e4.get()
    X = np.array(pd.read_csv(xpath, index_col=0))
    y = np.array(pd.read_csv(ypath, index_col=0))
    if var == 'DEA':
        dea = DEA(X, y)
        rs = dea.fit()
    elif var == 'BCC_DEA':
        deabcc = BCC_DEA(X, y, eps)
        rs = deabcc.fit()
    else:
        deaSBM = SBM_DEA(X, y)
        rs = deaSBM.fit()

    result = pd.DataFrame(rs)
    result.to_csv(savepath)
    messagebox.showinfo(title='Hi', message='DEA运行完成！')

b1 = Button(root, text='run DEA', command=runDEA)
b1.pack()

root.mainloop()

