

import time
import traceback
import logging
import os
log = logging.getLogger(__name__)
from pycqed.analysis import analysis_toolbox as a_tools
a_tools.datadir = 'Q:\\USERS\\nathan\\data\\xld'

class AnalysisDaemon:
    def __init__(self, t_start=None, start=True):
        self.t_start = t_start
        self.last_ts = None
        self.poll_interval = 5  # seconds
        self.errs = []
        self.job_errs = []
        if start:
            self.start()

    def start(self):
        self.last_ts = a_tools.latest_data(older_than=self.t_start,
                                           return_path=False,
                                           return_timestamp=True, n_matches=1)[0]
        self.run()

    def run(self):
        try:
            while (True):
                self.check_job()
                for i in range(self.poll_interval):
                    time.sleep(1)
        except KeyboardInterrupt as e:
            pass
        except Exception as e:
            log.error(e)
            self.errs.append(traceback.format_exc())
            self.run()

    def check_job(self):
        try:
            timestamps, folders = a_tools.latest_data(newer_than=self.last_ts,
                                                      raise_exc=False,
                                                      return_timestamp=True,
                                                      n_matches=1000)
        except ValueError as e:
            return  # could not find any timestamps matching criteria
        log.info(f"Searching in: {timestamps}")
        for folder, ts in zip(folders, timestamps):
            jobs_in_folder = []
            for file in os.listdir(folder):
                if file.endswith(".job"):
                    jobs_in_folder.append(os.path.join(folder, file))
            if len(jobs_in_folder) > 0:
                log.info(f"Found {len(jobs_in_folder)} jobs in {ts}")

            for filename in jobs_in_folder:
                if os.path.isfile(filename):
                    time.sleep(1)  # wait to make sure that the file is fully written

                    job = self.read_job(filename)
                    errl = len(self.job_errs)
                    self.run_job(job, ts)
                    if errl == len(self.errs):
                        os.rename(filename, filename + '.done')
                    else:
                        os.rename(filename, filename + '.failed')
                        new_errors = self.errs[errl:]
                        self.write_to_job(filename + '.failed', new_errors)
                    self.last_ts = ts

    def read_job(self, filename):
        job_file = open(filename, 'r')
        job = "".join(job_file.readlines())
        job_file.close()
        return job

    def write_to_job(self, filename, new_lines):
        job_file = open(filename, 'r+')
        job_file.write("\n")
        job_file.write("".join(new_lines))
        job_file.close()

    def run_job(self, job, ts):
        try:
            exec(job)
        except Exception as e:
            log.error(f"Error in job: {job}:\n{e}")
            self.job_errs.append(traceback.format_exc())

if __name__ == "__main__":
    ad = AnalysisDaemon(t_start=None)
