import csv
import re
import datetime

def sort_list(unsorted_list):
    temp = list()
    for i in range(len(unsorted_list)):
        date_time = unsorted_list[i]
        date_time_token = date_time.split(" ")
        date_token = date_time_token[0].split("/")
        date_token.reverse()
        temp_date_time = "/".join(date_token)
        temp_date_time = temp_date_time + " " + date_time_token[1]
        temp.append(temp_date_time)
    temp.sort()

    temp_1 = list()
    for i in range(len(temp)):
        date_time = temp[i]
        date_time_token = date_time.split(" ")
        date_token = date_time_token[0].split("/")
        date_token.reverse()
        temp_date_time = "/".join(date_token)
        temp_date_time = temp_date_time + " " + date_time_token[1]
        temp_1.append(temp_date_time)
    unsorted_list = temp_1

    return unsorted_list


#import data from heart_rate_only.csv, which should have date and bpm values
with open('heart_rate_trial.csv', newline='') as f:
    reader = csv.reader(f)
    heart_data = list(reader)


#Use flags to average out all BPM values recorded within the same minute.
complete_data_list = []
time_flag1 = heart_data[0][0]
numlist = []
heart_time_list=[]
for i in heart_data:
    time_flag2 = i[0]
    if time_flag2 == time_flag1:
       bpm_val = i[1]
       notcleaned_bpm = re.findall(r'\d+', bpm_val)
       numlist.append(int(notcleaned_bpm[0]))
    elif time_flag2 != time_flag1:
       complete_data_list.append(round(sum(numlist)/len(numlist),2))
       heart_time_list.append(time_flag1)
       numlist = []
       time_flag1 = i[0]
       bpm_val = i[1]
       notcleaned_bpm = re.findall(r'\d+', bpm_val)
       numlist.append(int(notcleaned_bpm[0]))
complete_data_list.append(round(sum(numlist)/len(numlist),2))
heart_time_list.append(time_flag1)


#Remove all values that are not shared 
with open('distance_trial.csv', newline='') as f:
    reader = csv.reader(f)
    distance_data = list(reader)

heart_time_list = sort_list(heart_time_list)

print("htl is", heart_time_list)

distance_time_list=[]
for i in distance_data:
    distance_time_list.append(i[0])
combined_heart_values = []
combined_distance_values = []

distance_time_list = sort_list(distance_time_list)

print("DTL is", distance_time_list)

heart_time_list = sort_list(heart_time_list)

print("HTL is", heart_time_list)

combined_time_list = list(set(distance_time_list).intersection(heart_time_list))   #####Converting to set to grab unique dates messes up order

combined_time_list = sort_list(combined_time_list)

print("CTL is", combined_time_list)
for i in heart_data:
    for j in combined_time_list:
        if i[0] == j:
            combined_heart_values.append(i[1])
for i in distance_data:
    for j in combined_time_list:
        if i[0] == j:
            combined_distance_values.append(i[1])
print("CHV is", combined_heart_values)
print("CDV is", combined_distance_values)


#Rewrite complete_data_list to trialfile.csv
file = open('trialfile.csv', 'w+', newline = "")
writer = csv.writer(file)
for w in range(len(combined_time_list)):
    writer.writerow([combined_time_list[w],combined_heart_values[w],combined_distance_values[w]])
file.close()
