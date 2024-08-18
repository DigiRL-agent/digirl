from digirl.misc import colorful_print
import threading
import os
import torch
import time

def remote_collect_trajectories(save_path, 
                                worker_temp_path, 
                                worker_run_path, 
                                worker_ips, 
                                worker_username, 
                                trainer):
    # add all workers into known hosts if not already
    colorful_print("Adding all workers to known hosts", fg='green')
    for worker_ip in worker_ips:
        print("worker_ip", worker_ip)
        os.system(f"ssh-keyscan -H {worker_ip} >> ~/.ssh/known_hosts")
    # kill all processes
    for worker_ip in worker_ips:
        os.system(f"ssh {worker_username}@{worker_ip} 'pkill -U {worker_username}'")
    time.sleep(5)
    for worker_ip in worker_ips:
        os.system(f"ssh {worker_username}@{worker_ip} 'skill -u {worker_username}'")
    time.sleep(5)
    
    # copying the agent to all remote workers
    # save the current trainer, NO MATTER it's zero-shot or offline or online
    colorful_print("Saving the current trainer", fg='green')
    trainer.save(os.path.join(save_path, "trainer_current.pt"))
    colorful_print("Copying the current trainer to all workers", fg='green')

    command = f"rm -rf {worker_temp_path} && mkdir -p {worker_temp_path} && exit"
    # parallely execute this command in all remote workser and wait for the command to finish
    threads = []
    colorful_print("Starting all trajectory collections", fg='green')
    for worker_ip in worker_ips:
        t = threading.Thread(target=os.system, args=(f"""ssh -tt {worker_username}@{worker_ip} << EOF 
{command}
EOF
""",))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
        colorful_print("Trajectory collection finished", fg='green')

    for worker_ip in worker_ips:
        command = f"scp -r {save_path}/trainer_current.pt {worker_username}@{worker_ip}:{worker_temp_path}"
        os.system(command)

    command = f"conda activate digirl && cd {worker_run_path} && python run.py --config-path config/multimachine --config-name worker && exit"
    for worker_ip in worker_ips:
        t = threading.Thread(target=os.system, args=(f"""ssh -tt {worker_username}@{worker_ip} << EOF 
{command}
EOF
""",))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
        colorful_print("Trajectory collection finished", fg='green')
    
    for worker_ip in worker_ips:
        os.system(f"scp {worker_username}@{worker_ip}:{worker_temp_path}/trajectories.pt {save_path}/{worker_ip}")
    # wait for all trajs to be scp'ed to this host machine
    while True:
        if all([os.path.exists(f"{save_path}/{worker_ip}") for worker_ip in worker_ips]):
            break
        time.sleep(5)

    # load all trajs in the remote machine
    trajectories_list = [torch.load(f"{save_path}/{worker_ip}") for worker_ip in worker_ips]
    # aggregate all trajs
    trajectories = []
    for traj_list in trajectories_list:
        for traj in traj_list:
            trajectories.append(traj)
    return trajectories
