from snake import Snake
snake = Snake()

snake.display()
#actions = [2,2,3,3,0,0,1,1]
actions = {'a':0, 'w':1, 'd':2, 's':3}
for i in range(20):
    action_str = raw_input('Move with: a,w,d,s + enter')
    action = actions[action_str]
    snake.play(action)
    snake.display()
