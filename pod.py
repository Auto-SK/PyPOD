# -*- coding: utf-8 -*-

"""
    @project: MAPOD
    @version: v1.0.0
    @file: pod.py
    @brief: POD Class
    @software: PyCharm
    @author: Kai Sun
    @email: autosunkai@gmail.com
    @date: 2020/11/28
    @updated: 2020/11/28
"""

import pandas as pd
import numpy as np
from scipy.stats import norm


class PoD:
    """ PoD Class """

    def __init__(self, input_filename, output_filename):
        self.input_filename = input_filename
        self.output_filename = output_filename
        self.x_log = True
        self.y_log = True
        self.a, self.a_hat = self.read_data()
        self.a_hat_th = np.mean(self.a_hat[np.where(self.a == np.min(self.a))])

    def read_data(self):
        """ Read data from csv file """
        filetype = self.input_filename.split('.')[1]
        if filetype == 'xlsx' or 'xls':
            df = pd.read_excel(self.input_filename)
        elif filetype == 'csv':
            df = pd.read_csv(self.input_filename)
        else:
            raise RuntimeError('File type is neither Excel nor CSV.')
        data = df.values
        a = data[:, 1]
        a_hat = data[:, 2]
        return a, a_hat

    def save_data(self):
        mu, sigma, pcov = self.pod_calc()
        p = np.linspace(0.001, 0.999, 200)
        a_pod = np.exp(norm.ppf(p, mu, sigma))
        wp = norm.ppf(p, 0, 1)
        a_pod_95 = self.pod_ci(pcov, a_pod, wp)
        with open(self.output_filename, 'w') as f1:
            np.savetxt(f1, (np.vstack((a_pod, p))).T, fmt='%9.6e')

    def regression(self):
        """ Liner regression """
        if self.x_log:
            x = np.log(self.a)
        else:
            x = self.a
        if self.y_log:
            y = np.log(self.a_hat)
        else:
            y = self.a_hat
        beta1 = sum((x - np.mean(x)) * (y - np.mean(y))) / sum((x - np.mean(x)) * (x - np.mean(x)))
        beta0 = np.mean(y) - beta1 * np.mean(x)
        tau = np.sqrt(sum((y - (beta0 + beta1 * x)) * (y - (beta0 + beta1 * x))) / len(x))
        return beta0, beta1, tau

    def pod_calc(self):
        """ Calculate POD """
        beta0, beta1, tau = self.regression()
        mu = (np.log(self.a_hat_th) - beta0) / beta1
        sigma = tau / beta1
        phi = np.array([[1, 0], [mu, sigma], [0, -1]]) * (-1) / beta1
        cov_lr = self.cov_para3(tau)
        pcov = np.matmul(np.matmul(phi.transpose(), cov_lr), phi)
        return mu, sigma, pcov

    def pod_para(self):
        """ Parameters of POD """
        mu, sigma, pcov = self.pod_calc()
        a_50 = np.exp(norm.ppf(0.5, mu, sigma))
        a_90 = np.exp(norm.ppf(0.9, mu, sigma))
        wp = norm.ppf(0.9, 0, 1)
        a_90_95 = self.pod_ci(pcov, a_90, wp)
        return a_50, a_90, a_90_95

    def cov_para1(self, tau):
        var0 = len(self.a) / tau ** 2
        var1 = sum(np.log(self.a) * np.log(self.a)) / tau ** 2
        cov_para = sum(np.log(self.a)) / tau ** 2
        return var0, var1, cov_para

    def cov_para2(self, tau):
        var0, var1, cov_para = self.cov_para1(tau)
        FIM = np.array([[var0, cov_para], [cov_para, var1]])
        pcov = np.linalg.inv(FIM)
        return pcov

    def cov_para3(self, tau):
        var0, var1, cov_para = self.cov_para1(tau)
        var2 = 2 * len(self.a) / tau ** 2
        FIM = np.array([[var0, cov_para, 0], [cov_para, var1, 0], [0, 0, var2]])
        cov_lr = np.linalg.inv(FIM)
        return cov_lr

    def pod_ci(self, pcov, a, wp):
        sd = np.sqrt(pcov[0, 0] + wp * wp * pcov[1, 1] + 2 * wp * pcov[0, 1])
        a_95 = np.exp(np.log(a) + 1.645 * sd)
        return a_95
