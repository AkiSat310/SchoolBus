import requests

token = 'Is2EQ9c8lVbVjHrjfM7QxULNrEfIaOoFz1s9UkTKWb0'
api = 'https://notify-api.line.me/api/notify'
message = '\n\nLIMIT CHECK'

headers = {'Authorization': 'Bearer'+' '+token}
data = {'message': message}
resp = requests.post(api,headers=headers,data=data)

print(message)

# APIコール残
ratelimit = resp.headers.get("X-RateLimit-Limit") # max API call
ratelimit_remaining = resp.headers.get("X-RateLimit-Remaining") # API call remaining

print('API call remaining : {} / {}'.format(ratelimit_remaining,ratelimit))

# 画像うp残
ratelimit_image = resp.headers.get("X-RateLimit-ImageLimit") # max image upload at 1hour
ratelimit_image_remaining = resp.headers.get("X-RateLimit-ImageRemaining") # image upload remaining

print('Image upload remaining : {} / {} by an hour'.format(ratelimit_image_remaining,ratelimit_image))

# リセット時間
ratelimit_reset = resp.headers.get("X-RateLimit-Reset") # reset time UTC
print('Reset time UTC : {}'.format(ratelimit_reset))

# ステータスコード
print('HTTP status code : {}'.format(resp.status_code)) # HTTP status code