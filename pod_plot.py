# -*- coding: utf-8 -*-

"""
    @project: Mesh Tube
    @version: MAPOD
    @file: pod_plot.py
    @brief: 
    @software: PyCharm
    @author: Kai Sun
    @email: autosunkai@gmail.com
    @date: 2020/11/29
    @updated: 2020/11/29
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from pod import PoD


class PoDPlot(PoD):
    def plot_raw(self):
        """ Plot raw data scatter """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        fig.suptitle(r'Four Possible a vs. $\hat{a}$ Models')
        csfont = {'fontname': 'Times New Roman', 'fontsize': 16}

        ax1.plot(self.a, self.a_hat, 'ks', markersize=0.5)
        ax1.set_xlabel('Size, a (mm) \n (a)')
        ax1.set_ylabel(r'Response, $\hat{a}$ (mV)')

        ax2.plot(self.a, self.a_hat, 'ks', markersize=0.5)
        ax2.set_xscale('log')
        # ax2.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        # ax2.set_xticklabels(['$10^{-1}$', 2, 3, 4, 5, 6, 7, 8, 9, '$10^0$'])
        ax2.set_xlabel('Size, $\log(a)$ (mm) \n (b)')
        ax2.set_ylabel(r'Response, $\hat{a}$ (mV)')

        ax3.plot(self.a, self.a_hat, 'ks', markersize=0.5)
        ax3.set_yscale('log')
        # ax3.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        # ax3.set_yticklabels(['$10^{-1}$', 2, 3, 4, 5, 6, 7, 8, 9, '$10^0$'])
        ax3.set_xlabel('Size, a (mm) \n (c)')
        ax3.set_ylabel(r'Response, $\log(\hat{a})$ (mV)')

        ax4.plot(self.a, self.a_hat, 'ks', markersize=0.5)
        ax4.set_xscale('log')
        ax4.set_yscale('log')
        # ax4.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        # ax4.set_xticklabels(['$10^{-1}$', 2, 3, 4, 5, 6, 7, 8, 9, '$10^0$'])
        # ax4.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        # ax4.set_yticklabels(['$10^{-1}$', 2, 3, 4, 5, 6, 7, 8, 9, '$10^0$'])
        ax4.set_xlabel('Size, $\log(a)$ (mm) \n (d)')
        ax4.set_ylabel(r'Response, $\log(\hat{a})$ (mV)')

        # plt.tight_layout()
        plt.show()

    def plot_reg(self):
        """ Plot regression curve """
        beta0, beta1, tau = self.regression()
        pcov = self.cov_para2(tau)
        x_min = np.log(min(self.a))
        x_max = np.log(max(self.a))
        x_lin = np.linspace(x_min, x_max, 100)
        y_lin = beta0 + beta1 * x_lin

        var_y = pcov[0, 0] + 2 * x_lin * pcov[0, 1] + x_lin * x_lin * pcov[1, 1]
        var_total = var_y + tau ** 2

        y_lin_lb = beta0 + beta1 * x_lin - 1.645 * np.sqrt(var_y)
        y_lin_ub = beta0 + beta1 * x_lin + 1.645 * np.sqrt(var_y)
        y_lin_lb_total = beta0 + beta1 * x_lin - 1.645 * np.sqrt(var_total)
        y_lin_ub_total = beta0 + beta1 * x_lin + 1.645 * np.sqrt(var_total)

        plt.figure()
        fig, ax = plt.subplots()
        ax.plot(self.a, self.a_hat, 'ks', markersize=0.5)
        ax.plot(np.exp(x_lin), np.exp(y_lin), 'k', linewidth=1)
        ax.plot(np.exp(x_lin), np.exp(y_lin_lb), 'b--', linewidth=0.5)
        ax.plot(np.exp(x_lin), np.exp(y_lin_ub), 'b--', linewidth=0.5)
        ax.plot(np.exp(x_lin), np.exp(y_lin_lb_total), 'b--', linewidth=0.5)
        ax.plot(np.exp(x_lin), np.exp(y_lin_ub_total), 'b--', linewidth=0.5)
        csfont = {'fontname': 'Times New Roman', 'fontsize': 16}
        ax.set_xlabel("Size, a (mm)", **csfont)
        ax.set_ylabel(r"Response, $\hat{a}$ (mV)", **csfont)
        ax.set_yscale('log')
        ax.set_xscale('log')

        #   ax.set_xlim([0.09, 1.02])
        #   ax.set_ylim([0.009, 101])

        for tick in ax.get_xticklabels():
            tick.set_fontname("Times New Roman")
            tick.set_fontsize(10)
        for tick in ax.get_yticklabels():
            tick.set_fontname("Times New Roman")
            tick.set_fontsize(10)

        text = "Key parameters: \n $ \\beta_0 $ = %f, \n $ \\beta_1 $ = %f, \n $ \\tau $ = %f" % (beta0, beta1, tau)

        ax.text(min(self.a), np.mean(self.a_hat), text, style='italic',
                fontsize=10, fontname='Times New Roman',
                bbox={'facecolor': 'white', 'alpha': 0.0, 'pad': 10})

        plt.show()

    def plot_res(self):
        """ Plot residual error """
        pass

    def plot_pod(self):
        """ Plot pod curve """
        mu, sigma, pcov = self.pod_calc()
        p = np.linspace(0.001, 0.999, 200)
        a_pod = np.exp(norm.ppf(p, mu, sigma))
        wp = norm.ppf(p, 0, 1)
        a_pod_95 = self.pod_ci(pcov, a_pod, wp)
        plt.figure()
        fig, ax = plt.subplots()
        csfont = {'fontname': 'Times New Roman', 'fontsize': 16}
        # ax.set_xscale('log')
        # ax.set_xlim([min(a_pod), max(a_pod)])
        ax.plot(a_pod, p, 'k', label='Mean POD', linewidth=1)
        ax.plot(a_pod_95, p, 'k--', label='95% Lower Bound', linewidth=0.5)
        ax.set_xlabel('Size, a (mm)', **csfont)
        ax.set_ylabel('Probability of Detection, POD | a', **csfont)
        ax.legend()
        # ax.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        # ax.set_xticklabels(['$10^{-1}$', 2, 3, 4, 5, 6, 7, 8, 9, '$10^0$'])
        for tick in ax.get_xticklabels():
            tick.set_fontname("Times New Roman")
            tick.set_fontsize(10)
        for tick in ax.get_yticklabels():
            tick.set_fontname("Times New Roman")
            tick.set_fontsize(10)

        text = "Key parameters: \n $ \\mu $ = %f, \n $ \\sigma $ = %f, \n covariance matrix: \n [%f %f \n ${ } { } { }$ %f %f]" % (
            mu, sigma, pcov[0, 0], pcov[0, 1], pcov[1, 0], pcov[1, 1])
        ax.text(min(a_pod), 0.5, text, style='italic',
                fontsize=10, fontname='Times New Roman',
                bbox={'facecolor': 'white', 'alpha': 0.0, 'pad': 10})
        plt.show()
