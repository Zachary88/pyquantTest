import jqdatasdk as jq

jq.auth("17620111577", "ZwC226178@#")

"""
# 获得账号信息
count = jq.get_query_count()
print(count)
"""
infos = jq.get_future_contracts("M")

print(infos)