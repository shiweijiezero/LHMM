node_id_lst_lst = [1, 2, 3, 4, 4]
node_dis_lst_lst = [4, 5, 6, 4, 4]

node_zip_lst = list(zip(node_id_lst_lst, node_dis_lst_lst))
print(node_zip_lst)
all_path_node = list(set(node_zip_lst))
print(all_path_node)
remoteInfoNodeId = "1,2,3*4,5,6"
node_id_lst_lst = list(map(lambda e: [int(i) for i in e.split(",")], remoteInfoNodeId.split("*")))
print(node_id_lst_lst)
node_dis_lst_lst = "7,8,9*10,11,12"
node_dis_lst_lst = list(map(lambda e: [float(i) for i in e.split(",")], node_dis_lst_lst.split("*")))
print(node_dis_lst_lst)
all_node_pair = list(map(lambda a, b: list(zip(a, b)), node_id_lst_lst, node_dis_lst_lst))
print(all_node_pair)
