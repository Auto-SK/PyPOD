# -*- coding: utf-8 -*-

"""
    @project: MAPOD
    @version: v1.0.0
    @file: main.py
    @brief: main file
    @software: PyCharm
    @author: Kai Sun
    @email: autosunkai@gmail.com
    @date: 2020/11/29
    @updated: 2020/11/29
"""

from pod import PoD


def run():
    pod = PoD('TMR.xlsx', 'pod.txt')
    pod.save_data()


if __name__ == '__main__':
    run()
