# -*- coding: utf8 -*-

import game.sound.echo
game.sound.echo.echo_test()

# 다음과 같이 사용하는 것은 불가능하다. (Java는 가능하지만)
# import game
# game.sound.echo.echo_test()
# import game 을 수행하면 game 디렉토리의 모듈 또는 game 디렉토리의 __init__.py 에 정의된 것들만 참조가 가능하다.

# import game.sound.echo.echo_test
# import a.b.c 처럼 import 할 때 가장 마지막 항목인 c는 모듈 또는 패키지여 야만 한다.

from game.sound import echo
echo.echo_test()

from game.sound.echo import echo_test
echo_test()