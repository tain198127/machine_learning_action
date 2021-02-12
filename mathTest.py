import math
import sys
aaa = math.fabs(-2147483648)
print(aaa)
print(sys.maxsize)

absMin = -sys.maxsize-1;
print('{}:{}'.format('sysMin',absMin))
print('{}:{}'.format('absSysMin',abs(absMin)))


bbb = abs(-2147483648)
print(bbb)