import zipfile
import csv
import io
from datetime import datetime
import sys

# will hold vitals
class Data:
    def __init__(self, heartRate):
        self.heartRates = [ heartRate ]
        self.steps = []

    def add_heartrate(self, heartRate):
        self.heartRates.append(heartRate)

    def add_steps(self, steps):
        self.steps.append(steps)

    def get_heartrate(self):
        return round(sum(self.heartRates) / len(self.heartRates))

    def get_steps(self):
        if len(self.steps) == 0:
            return 0

        return round(sum(self.steps) / len(self.steps))

def OpenZipFile(zipFile, file):
    with zipfile.ZipFile(zipFile) as zf:
        return io.TextIOWrapper(zf.open(file), encoding="utf-8")

# get datetime to minute accuracy (zeroing out the seconds)
def GetDateTimeToMinute(strDateTime):
    dt = datetime.strptime(strDateTime, '%m/%d/%Y %I:%M:%S %p')
    return datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute)

def ReadHeartRateFromCsv(file):
    reader = csv.reader(file, delimiter=',')
    count = 1
    print('Processing HeartRate Data')
    for row in reader:
        if row[0] == 'Id':  # don't process header row
            continue

        if(count % 10000 == 0):
            print('Processed', count, 'rows')

        count += 1
        try:
            dt = GetDateTimeToMinute(row[1])
            if dict.get(dt) is None:
                data = Data(int(row[2]))
                dict[dt] = data
            else:
                dict[dt].add_heartrate(int(row[2]))
        except: # something went wrong, just skip this row
            print('something went wrong', sys.exc_info()[0], sys.exc_info()[1])
            continue

    file.close()

def ReadStepsFromCsv(file):
    reader = csv.reader(file, delimiter=',')
    count = 1
    print('Processing Steps Data')
    for row in reader:
        if row[0] == 'Id':  # don't process header row
            continue

        if(count % 10000 == 0):
            print('Processed', count, 'rows')

        count += 1
        try:
            dt = GetDateTimeToMinute(row[1])
            if dict.get(dt) is None: # no heartrate record for this time, just skip
                continue
            else:
                dict[dt].add_steps(int(row[2]))
        except: # something went wrong, just skip this row
            print('something went wrong', sys.exc_info()[0], sys.exc_info()[1])
            continue

    file.close()

def WriteCsv(outFile):
    with open(outFile, 'w', newline = "") as file:
        writer = csv.writer(file)
        dateTimes = [*dict] # unpack dictionary keys to list
        dateTimes.sort()
        for dt in dateTimes:
            writer.writerow([dt.strftime('%Y/%m/%d %H:%M:%S'),dict[dt].get_heartrate(),dict[dt].get_steps()])

# dictionary of (datetime, Data)
dict = {}

zipArchive = 'archive.zip'
heartRateFile = 'Fitabase Data 4.12.16-5.12.16/heartrate_seconds_merged.csv'
stepsFile = 'Fitabase Data 4.12.16-5.12.16/minuteStepsNarrow_merged.csv'
outFile = 'vitals_data.csv'

zipHeart = OpenZipFile(zipArchive, heartRateFile)
ReadHeartRateFromCsv(zipHeart)

zipSteps = OpenZipFile(zipArchive, stepsFile)
ReadStepsFromCsv(zipSteps)

WriteCsv(outFile)
