import os
import sys

ans = 0
a = 1
i = 1
b = 20230610
while (ans <b):
    i = i + 1
    a = a + i
    ans = ans + a

if (ans==b):
    print(i)
if (ans > b):
    print(i - 1)

# 请在此输入您的代码