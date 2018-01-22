# http://blog.bradlucas.com/posts/2017-06-03-yahoo-finance-quote-download-python/
# https://github.com/bradlucas/get-yahoo-quotes-python/blob/master/get-yahoo-quotes.py
# !/usr/bin/env python

"""
get-yahoo-quotes.py:  Script to download Yahoo historical quotes using the new cookie authenticated site.
 Usage: get-yahoo-quotes SYMBOL
 History
 06-03-2017 : Created script
"""

__author__ = "Brad Luicas"
__copyright__ = "Copyright 2017, Brad Lucas"
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brad Lucas"
__email__ = "brad@beaconhill.com"
__status__ = "Production"

import re
import sys
import time
import datetime
import requests


def split_crumb_store(v):
    return v.split(':')[2].strip('"')


def find_crumb_store(lines):
    # Looking for
    # ,"CrumbStore":{"crumb":"9q.A4D1c.b9

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    #########################new code##############################
    pattern = re.compile('CrumbStore":{"crumb":"(.*?)"}')
    for line in lines:
        m = pattern.match(line)
        if m is not None:
            crumb = m.groupdict()['crumb']
            crumb = crumb.replace(u'\\u002F', '/')
            return crumb
    #########################new code##############################
    #for l in lines: ------------ previous code
    #    if re.findall(r'CrumbStore', l):
    #        return l


def get_cookie_value(r):
    return {'B': r.cookies['B']}


def get_page_data(symbol):
    url = "https://finance.yahoo.com/quote/%s/?p=%s" % (symbol, symbol)
    r = requests.get(url)

    cookie = get_cookie_value(r)

    # Code to replace possible \u002F value
    # ,"CrumbStore":{"crumb":"FWP\u002F5EFll3U"
    # FWP\u002F5EFll3U
    lines = r.content.decode('unicode-escape').strip().replace('}', '\n')
    return cookie, lines.split('\n')


def get_cookie_crumb(symbol):
    cookie, lines = get_page_data(symbol)
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    #########################new code##############################
    crumb = find_crumb_store(lines) # do not use "split_crumb_store"
    #########################new code##############################
    return cookie, crumb


#
#
# Here, you can change your saving directory
#
#
def get_data(symbol, start_date, end_date, cookie, crumb):
    filename = '%s.csv' % (symbol)
    url = "https://query1.finance.yahoo.com/v7/finance/download/%s?period1=%s&period2=%s&interval=1d&events=history&crumb=%s" % (
    symbol, start_date, end_date, crumb)
    response = requests.get(url, cookies=cookie)
    with open(filename, 'wb') as handle:
        for block in response.iter_content(1024):
            handle.write(block)


def get_now_epoch():
    # @see https://www.linuxquestions.org/questions/programming-9/python-datetime-to-epoch-4175520007/#post5244109
    return int(time.time())


def download_quotes(symbol):
    start_date = 0
    end_date = get_now_epoch()
    cookie, crumb = get_cookie_crumb(symbol)
    get_data(symbol, start_date, end_date, cookie, crumb)


####################################################################################################################


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#########################new code##############################
def Russell(Number):
    from bs4 import BeautifulSoup
    from selenium import webdriver
    import os
    import platform
    output = []
    if platform.system() == 'Windows':
        pwd = os.getcwd() + '\chromedriver'
    else:
        pwd = os.getcwd() + '/chromedriver'
    for page_iter in range(1, int(Number/100)+1):
        Index = 'https://www.barchart.com/stocks/indices/russell/russell%s?page=%d' % (Number, page_iter)
        driver = webdriver.Chrome(pwd)
        driver.get(Index)
        soup = BeautifulSoup(driver.page_source, 'lxml')
        TagList = soup.findAll('tr')
        for Tag in TagList:
            ticker = Tag.get("data-current-symbol")
            if str(ticker) != 'None':
                output.append(ticker)
    return output
ticker = Russell(1000)
#########################new code##############################


print(len(ticker))
i = 1
#########################new code##############################
for symbol in ticker:
    fail = 1
    while fail > 0:
        try:
            download_quotes(symbol)
            print(i, symbol)
            fail = 0
            i += 1
        except KeyError:
            time.sleep(0.1)
            print('#Fail = %d' % fail)
            fail += 1
#########################new code##############################