# import torch
# import numpy as np
# import time
# import json

# def _build_edge_index_from_ego_list(ego_list):
#     if len(ego_list) == 0:
#         return torch.empty((2, 0), dtype=torch.long)
#     return torch.tensor(np.concatenate(ego_list, axis=0), dtype=torch.long).t().contiguous()

# def train(env, num_episodes, actors, critic,
#           optimizers_actors, optimizer_critic, gamma=0.99,
#           max_steps_per_episode=200, device=None, verbose=True,
#           save_dir=None, save_every=10, plot_rewards_fn=None,plot_task_stats=None, plot_values_fn=None,save_models_fn=None):

#     episode_rewards = []
#     episode_task_stats = [] 
#     episode_value_means = [] 

#     if device is None:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     for a in actors.values():
#         a.to(device)
#     critic.to(device)

#     # entropy_coef = 0.05  # example; adjust as needed
    
#     try:
#         for episode in range(num_episodes):
#             entropy_coef = max(0.01, 0.05 * (0.995 ** episode))
#             print(f"=== Episode {episode+1}/{num_episodes} ===")
            
#             obs, _ = env.reset()
#             ego_graphs, attribute_matrix = obs

#             episode_values = []

#             done = False
#             episode_reward = 0.0
#             start = time.time()

#             step = 0
#             while (not done) and step < max_steps_per_episode:
#                 # print(f" Episode {episode+1} Step {step+1} ---------------------")
#                 actions = {}
#                 # We'll fill log_probs AFTER env.step based on resolved assignments
#                 log_probs = {}
#                 entropies = {}

#                 # For later computing executed action's log_prob
#                 actor_candidates = {}     # rid -> tensor of node indices
#                 actor_task_logits = {}    # rid -> tensor of logits aligned with actor_candidates[rid]

#                 value_preds = {}

#                 x = torch.tensor(attribute_matrix, dtype=torch.float, device=device)
#                 edge_index_cache = {}  # cache edge_index per robot to avoid repeated builds

#                 # ---------- ACTOR (produce top-2 proposals and store logits) ----------
#                 with torch.no_grad():
#                     for rid, ego_list in ego_graphs.items():
#                         # if rid<1:
#                             # print(f" Robot {rid} has {len(ego_list), ego_list} ego edges.")
#                         if len(ego_list) == 0:
#                             actions[rid] = []
#                             continue

#                         # Build edge_index for actor (ego graph)
#                         edge_index = _build_edge_index_from_ego_list(ego_list).to(device)
#                         edge_index_cache[rid] = edge_index

#                         # Actor forward (per-robot policy uses full node features x and ego edge_index)
#                         logits = actors[rid](x, edge_index)  # per-node scores, indexed by global node idx

#                         # Determine which task nodes are actually present in ego_list.
#                         nodes_present = torch.unique(edge_index).cpu().numpy().tolist()
#                         # filter to task nodes in current attribute matrix
#                         task_nodes_present = [n for n in nodes_present if n >= env.n_robots and n < env.n_robots + env.n_tasks]
#                         if len(task_nodes_present) == 0:
#                             actions[rid] = []
#                             continue

#                         task_nodes_tensor = torch.tensor(task_nodes_present, dtype=torch.long, device=device)
#                         task_logits = logits[task_nodes_tensor]  # logits for these candidate nodes

#                         # store candidates & logits for later log_prob computation
#                         actor_candidates[rid] = task_nodes_tensor  # node indices
#                         actor_task_logits[rid] = task_logits       # corresponding logits

#                         # choose top-2 by logits to propose (resolve_conflicts can use second choice)
#                         # k = min(2, task_logits.size(0))
#                         # topk_indices = torch.topk(task_logits, k=k).indices.cpu().tolist()
#                         # top_nodes = [int(task_nodes_tensor[i].item()) for i in topk_indices]
#                         # if len(top_nodes) == 1:
#                         #     top_nodes.append(top_nodes[0])
#                         # actions[rid] = top_nodes
#                         dist = torch.distributions.Categorical(logits=task_logits)
#                         sample_idx = dist.sample()
#                         chosen_node = int(task_nodes_tensor[sample_idx].item())
#                         actions[rid] = [chosen_node]
#                         # Store log prob + entropy NOW (no need to reconstruct later)
#                         log_probs[rid] = dist.log_prob(sample_idx)
#                         entropies[rid] = dist.entropy()

#                 # ---------- CRITIC EVALUATION ----------
#                 # Build global edge_index by concatenating all ego lists if necessary.
#                 all_edge_list = []
#                 for ego_list in ego_graphs.values():
#                     if len(ego_list) > 0:
#                         all_edge_list.append(np.concatenate(ego_list, axis=0))
#                 if all_edge_list:
#                     global_edge_index = torch.tensor(np.concatenate(all_edge_list, axis=0), dtype=torch.long).t().contiguous().to(device)
#                 else:
#                     global_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

#                 # Call critic ONCE per step. Provide num_robots so critic returns per-robot vector when configured.
#                 critic_out = critic(x, global_edge_index, batch=None, num_robots=env.n_robots)

#                 # Normalize critic_out handling:
#                 for rid in range(env.n_robots):
#                     if isinstance(critic_out, torch.Tensor) and critic_out.dim() > 0 and critic_out.numel() > 1:
#                         v = critic_out[rid]
#                     else:
#                         v = critic_out
#                     value_preds[rid] = v.squeeze()
#                     # --- Track mean value prediction this step ---
#                 if isinstance(critic_out, torch.Tensor):
#                     episode_values.append(critic_out.detach().mean().cpu().item())
#                 else:
#                     episode_values.append(float(critic_out))

#                 # ---------- ENV STEP ----------
#                 next_obs, reward, done, truncated, info_reward, info = env.step(actions, assignment_interval=5)
#                 ego_graphs_next, attribute_matrix_next = next_obs
# #------------
#                 # # Get resolved assignments actually applied by the env (rid -> task_identifier)
#                 # resolved = info.get("resolved_assignments", {})

#                 # # For every robot that actually got an assignment, compute the log_prob under actor
#                 # for rid, assigned_task_id in resolved.items():
#                 #     # assigned_task_id might be a node index (n_robots + task_idx) or a unique task id.
#                 #     # Our actor_candidates use node indices. Prefer node-index path.
#                 #     # If assigned_task_id is a float/unique id, try to map to node index via env.taskid_to_task
#                 #     node_assigned = assigned_task_id
#                 #     # Map unique id to node index if necessary
#                 #     if assigned_task_id not in actor_candidates.get(rid, []):
#                 #         # try mapping from task id to node index
#                 #         if assigned_task_id in env.taskid_to_task:
#                 #             # find position of task in self.tasks and compute node index
#                 #             try:
#                 #                 t_idx = [t.id for t in env.tasks].index(assigned_task_id)
#                 #                 node_assigned = env.n_robots + t_idx
#                 #             except ValueError:
#                 #                 node_assigned = assigned_task_id  # leave as is
#                 #     # Now compute log_prob if node_assigned is in actor_candidates
#                 #     if rid in actor_candidates:
#                 #         candidates = actor_candidates[rid]
#                 #         logits = actor_task_logits[rid]
#                 #         # find position of node_assigned in candidates
#                 #         matches = (candidates == node_assigned).nonzero(as_tuple=True)[0]
#                 #         if matches.numel() > 0:
#                 #             pos = matches[0].item()
#                 #             logp_all = torch.log_softmax(logits, dim=-1)
#                 #             log_probs[rid] = logp_all[pos]
#                 #             # compute entropy over this candidate set
#                 #             probs_all = torch.softmax(logits, dim=-1)
#                 #             entropies[rid] = - (probs_all * logp_all).sum()
#                 #         else:
#                 #             # assigned node not in candidate list (rare) -> skip
#                 #             pass
# #--------------

#                 # print(reward, info_reward,"reward")
#                 # accumulate rewards
#                 if isinstance(reward, dict):
#                     episode_reward += sum(reward.values())
#                 else:
#                     episode_reward += reward
#                 # print(episode_reward,"episode_reward")
#                 # ---------- Compute critic on next state for bootstrap ----------
#                 x_next = torch.tensor(attribute_matrix_next, dtype=torch.float, device=device)
#                 all_edge_list_next = []
#                 for ego_list in ego_graphs_next.values():
#                     if len(ego_list) > 0:
#                         all_edge_list_next.append(np.concatenate(ego_list, axis=0))
#                 if all_edge_list_next:
#                     global_edge_index_next = torch.tensor(np.concatenate(all_edge_list_next, axis=0), dtype=torch.long).t().contiguous().to(device)
#                 else:
#                     global_edge_index_next = torch.empty((2, 0), dtype=torch.long, device=device)

#                 critic_out_next = critic(x_next, global_edge_index_next, batch=None, num_robots=env.n_robots)

#                 value_next = {}
#                 for rid in value_preds.keys():
#                     if isinstance(critic_out_next, torch.Tensor) and critic_out_next.dim() > 0 and critic_out_next.numel() > 1:
#                         v_next = critic_out_next[rid]
#                     else:
#                         v_next = critic_out_next
#                     value_next[rid] = v_next.squeeze()

#                 # ---------- ADVANTAGES & TD TARGETS ----------
#                 advantages = {}
#                 td_targets = {}
#                 adv_list = []
#                 adv_keys = []
#                 for rid, r in (reward.items() if isinstance(reward, dict) else enumerate([reward])):
#                     if rid not in value_preds:
#                         continue
#                     v_curr = value_preds[rid]
#                     v_next = value_next.get(rid, None)
#                     done_or_trunc = float(done or truncated)
#                     if v_next is not None:
#                         target = r + gamma * v_next.detach() * (1.0 - done_or_trunc)
#                     else:
#                         target = torch.tensor(r, dtype=v_curr.dtype, device=device)
#                     adv = (target - v_curr)
#                     advantages[rid] = adv
#                     td_targets[rid] = target

#                     # safe scalar extraction for normalization
#                     adv_det = adv.detach()
#                     if adv_det.numel() == 1:
#                         adv_list.append(adv_det.cpu().item())
#                     else:
#                         adv_list.append(float(adv_det.cpu().mean().item()))
#                     adv_keys.append(rid)

#                 # Normalize advantages
#                 # if len(adv_list) > 0:
#                 #     adv_arr = np.array(adv_list, dtype=np.float32)
#                 #     mean = adv_arr.mean()
#                 #     std = adv_arr.std() if adv_arr.std() > 1e-8 else 1.0
#                 #     for i, rid in enumerate(adv_keys):
#                 #         norm_val = (adv_arr[i] - mean) / std
#                 #         advantages[rid] = torch.tensor(norm_val, dtype=advantages[rid].dtype, device=device)

#                 # ---------- CRITIC UPDATE ----------
#                 if td_targets:
#                     optimizer_critic.zero_grad()
#                     critic_loss = sum(((td_targets[rid] - value_preds[rid]) ** 2).mean()
#                                     for rid in td_targets)
#                     critic_loss.backward()
#                     optimizer_critic.step()

#                     # ---------- ACTOR UPDATE ----------
#                     for rid in list(advantages.keys()):
#                         # only update if we computed a log_prob for executed action
#                         if rid not in log_probs:
#                             continue
#                         optimizers_actors[rid].zero_grad()
#                         adv = advantages[rid].detach()
#                         entropy_term = entropies.get(rid, torch.tensor(0.0, device=adv.device))
#                         actor_loss = -(log_probs[rid] * adv).mean() - entropy_coef * entropy_term.mean()
#                         actor_loss.backward()
#                         optimizers_actors[rid].step()

#                 # advance
#                 ego_graphs = ego_graphs_next
#                 attribute_matrix = attribute_matrix_next
#                 step +=1  
#             # =========================== EPISODE TASK STATISTICS=================================
#             current_time = env.time_count

#             obsolete = 0
#             never_picked = 0
#             completed = 0

#             for task in env.tasks:
#                 if task.is_droppedoff:
#                     completed += 1
#                 elif task.is_obsolete(current_time):
#                     obsolete += 1
#                     if not task.is_pickedup:
#                         never_picked += 1

#             episode_task_stats.append({
#                 "episode": episode + 1,
#                 "obsolete": obsolete,
#                 "never_picked": never_picked,
#                 "completed": completed
#             })
#             episode_rewards.append(episode_reward)
#             if len(episode_values) > 0:
#                 episode_value_means.append(float(np.mean(episode_values)))
#             else:
#                 episode_value_means.append(0.0)


#             # Save results every `save_every` episodes
#             if save_dir and save_every > 0 and (episode + 1) % save_every == 0:
#                 # print(f"Checkpoint: Saving results at episode {episode+1}...")
#                 if save_models_fn:
#                     save_models_fn(save_dir, actors, critic)
#                 if plot_rewards_fn:
#                     plot_rewards_fn(save_dir, episode_rewards)
#                 if "plot_task_stats" in globals():
#                     plot_task_stats(save_dir, episode_task_stats)

#                 with open(save_dir / "episode_rewards.json", "w") as f:
#                     json.dump([float(x) for x in episode_rewards], f)
#                 with open(save_dir / "episode_task_stats.json", "w") as f:
#                     json.dump(episode_task_stats, f, indent=2)

#                 with open(save_dir / "episode_values.json", "w") as f:
#                     json.dump(episode_value_means, f)

#     except KeyboardInterrupt:
#         print("\nTraining interrupted by user. Saving progress...")
#         if save_dir:
#             if save_models_fn:
#                 save_models_fn(save_dir, actors, critic)
#             if plot_rewards_fn:
#                 plot_rewards_fn(save_dir, episode_rewards)
#             if plot_values_fn:
#                 plot_values_fn(save_dir, episode_value_means)
#             with open(save_dir / "episode_rewards.json", "w") as f:
#                 json.dump([float(x) for x in episode_rewards], f)
#             with open(save_dir / "episode_task_stats.json", "w") as f:
#                     json.dump(episode_task_stats, f, indent=2)

#     return episode_rewards, episode_task_stats, episode_value_means

#             # if verbose:
#             #     print(f"Episode {episode+1}/{num_episodes} Reward={episode_reward:.2f} Time={time.time()-start:.2f}s")
#             # episode_rewards.append(episode_reward)

#         # return episode_rewards

import torch
import numpy as np
import time
import json


def _build_edge_index_from_ego_list(ego_list):
    if len(ego_list) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.as_tensor(
        np.concatenate(ego_list, axis=0),
        dtype=torch.long
    ).t().contiguous()


# def train(env, num_episodes, actors, critic,
#           optimizers_actors, optimizer_critic, gamma=0.99,
#           max_steps_per_episode=200, device=None, verbose=True,
#           save_dir=None, save_every=10,
#           plot_rewards_fn=None,
#           plot_task_stats=None,
#           plot_values_fn=None,
#           save_models_fn=None):

#     episode_rewards = []
#     episode_task_stats = []
#     episode_value_means = []

#     if device is None:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     for a in actors.values():
#         a.to(device)
#     critic.to(device)

#     try:
#         for episode in range(num_episodes):

#             entropy_coef = max(0.01, 0.05 * (0.995 ** episode))

#             print(f"=== Episode {episode+1}/{num_episodes} ===")

#             obs, _ = env.reset()
#             ego_graphs, attribute_matrix = obs

#             # Cache tensor version once
#             x = torch.as_tensor(attribute_matrix, dtype=torch.float32, device=device)

#             episode_values = []
#             done = False
#             episode_reward = 0.0
#             step = 0

#             while (not done) and step < max_steps_per_episode:

#                 actions = {}
#                 log_probs = {}
#                 entropies = {}
#                 value_preds = {}

#                 edge_index_cache = {}
#                 # ================= ACTOR =================
#                 for rid, ego_list in ego_graphs.items():

#                     if len(ego_list) == 0:
#                         actions[rid] = []
#                         continue

#                     edge_index = _build_edge_index_from_ego_list(ego_list).to(device)

#                     # Forward WITH grad
#                     logits = actors[rid](x, edge_index)

#                     nodes_present = torch.unique(edge_index)
#                     task_mask = (nodes_present >= env.n_robots) & \
#                                 (nodes_present < env.n_robots + env.n_tasks)

#                     task_nodes_tensor = nodes_present[task_mask]

#                     if task_nodes_tensor.numel() == 0:
#                         actions[rid] = []
#                         continue

#                     task_logits = logits[task_nodes_tensor]

#                     dist = torch.distributions.Categorical(logits=task_logits)

#                     # Sample action (sampling itself is non-differentiable anyway)
#                     sample_idx = dist.sample()

#                     chosen_node = int(task_nodes_tensor[sample_idx].item())

#                     actions[rid] = [chosen_node]

#                     # THESE MUST HAVE GRAD
#                     log_probs[rid] = dist.log_prob(sample_idx)
#                     entropies[rid] = dist.entropy()


#                 # ================= CRITIC =================
#                 all_edge_list = []
#                 for ego_list in ego_graphs.values():
#                     if len(ego_list) > 0:
#                         all_edge_list.append(np.concatenate(ego_list, axis=0))

#                 if all_edge_list:
#                     global_edge_index = torch.as_tensor(
#                         np.concatenate(all_edge_list, axis=0),
#                         dtype=torch.long,
#                         device=device
#                     ).t().contiguous()
#                 else:
#                     global_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

#                 critic_out = critic(x, global_edge_index, batch=None, num_robots=env.n_robots)

#                 for rid in range(env.n_robots):
#                     if isinstance(critic_out, torch.Tensor) and critic_out.numel() > 1:
#                         value_preds[rid] = critic_out[rid].squeeze()
#                     else:
#                         value_preds[rid] = critic_out.squeeze()

#                 if isinstance(critic_out, torch.Tensor):
#                     episode_values.append(critic_out.detach().mean().cpu().item())
#                 else:
#                     episode_values.append(float(critic_out))

#                 # ================= ENV STEP =================
#                 next_obs, reward, done, truncated, info_reward, info = env.step(
#                     actions, assignment_interval=5
#                 )

#                 ego_graphs_next, attribute_matrix_next = next_obs
#                 x_next = torch.as_tensor(attribute_matrix_next, dtype=torch.float32, device=device)

#                 if isinstance(reward, dict):
#                     episode_reward += sum(reward.values())
#                 else:
#                     episode_reward += reward

#                 # ================= NEXT VALUE =================
#                 all_edge_list_next = []
#                 for ego_list in ego_graphs_next.values():
#                     if len(ego_list) > 0:
#                         all_edge_list_next.append(np.concatenate(ego_list, axis=0))

#                 if all_edge_list_next:
#                     global_edge_index_next = torch.as_tensor(
#                         np.concatenate(all_edge_list_next, axis=0),
#                         dtype=torch.long,
#                         device=device
#                     ).t().contiguous()
#                 else:
#                     global_edge_index_next = torch.empty((2, 0), dtype=torch.long, device=device)

#                 critic_out_next = critic(x_next, global_edge_index_next,
#                                          batch=None, num_robots=env.n_robots)

#                 value_next = {}
#                 for rid in value_preds.keys():
#                     if isinstance(critic_out_next, torch.Tensor) and critic_out_next.numel() > 1:
#                         value_next[rid] = critic_out_next[rid].squeeze()
#                     else:
#                         value_next[rid] = critic_out_next.squeeze()

#                 # ================= ADVANTAGE =================
#                 advantages = {}
#                 td_targets = {}

#                 for rid, r in (reward.items() if isinstance(reward, dict)
#                                else enumerate([reward])):

#                     if rid not in value_preds:
#                         continue

#                     v_curr = value_preds[rid]
#                     v_next = value_next.get(rid, None)

#                     done_flag = float(done or truncated)

#                     if v_next is not None:
#                         target = r + gamma * v_next.detach() * (1 - done_flag)
#                     else:
#                         target = torch.as_tensor(r, dtype=v_curr.dtype, device=device)

#                     advantages[rid] = target - v_curr
#                     td_targets[rid] = target

#                 # ================= CRITIC UPDATE =================
#                 if td_targets:
#                     optimizer_critic.zero_grad()
#                     critic_loss = sum(
#                         ((td_targets[rid] - value_preds[rid]) ** 2).mean()
#                         for rid in td_targets
#                     )
#                     critic_loss.backward()
#                     optimizer_critic.step()

#                     # ================= ACTOR UPDATE =================
#                     for rid in advantages.keys():

#                         if rid not in log_probs:
#                             continue

#                         optimizers_actors[rid].zero_grad()

#                         adv = advantages[rid].detach()
#                         entropy_term = entropies.get(
#                             rid, torch.tensor(0.0, device=device)
#                         )

#                         actor_loss = -(log_probs[rid] * adv) \
#                                      - entropy_coef * entropy_term

#                         actor_loss.backward()
#                         optimizers_actors[rid].step()

#                 # advance
#                 ego_graphs = ego_graphs_next
#                 attribute_matrix = attribute_matrix_next
#                 x = x_next
#                 step += 1

#             # ================= EPISODE TASK STATS =================
#             current_time = env.time_count

#             obsolete = 0
#             never_picked = 0
#             completed = 0

#             for task in env.tasks:
#                 if task.is_droppedoff:
#                     completed += 1
#                 elif task.is_obsolete(current_time):
#                     obsolete += 1
#                     if not task.is_pickedup:
#                         never_picked += 1

#             episode_task_stats.append({
#                 "episode": episode + 1,
#                 "obsolete": obsolete,
#                 "never_picked": never_picked,
#                 "completed": completed
#             })

#             episode_rewards.append(episode_reward)

#             if len(episode_values) > 0:
#                 episode_value_means.append(float(np.mean(episode_values)))
#             else:
#                 episode_value_means.append(0.0)

#             # ================= SAVE =================
#             if save_dir and save_every > 0 and (episode + 1) % save_every == 0:

#                 if save_models_fn:
#                     save_models_fn(save_dir, actors, critic)

#                 if plot_rewards_fn:
#                     plot_rewards_fn(save_dir, episode_rewards)

#                 if plot_task_stats is not None:
#                     plot_task_stats(save_dir, episode_task_stats)

#                 if plot_values_fn is not None:
#                     plot_values_fn(save_dir, episode_value_means)

#                 with open(save_dir / "episode_rewards.json", "w") as f:
#                     json.dump([float(x) for x in episode_rewards], f)

#                 with open(save_dir / "episode_task_stats.json", "w") as f:
#                     json.dump(episode_task_stats, f, indent=2)

#                 with open(save_dir / "episode_values.json", "w") as f:
#                     json.dump(episode_value_means, f)

#     except KeyboardInterrupt:
#         print("\nTraining interrupted. Saving progress...")
#         if save_dir:
#             if save_models_fn:
#                 save_models_fn(save_dir, actors, critic)

#     return episode_rewards, episode_task_stats, episode_value_means


# def train(env, num_episodes, actors, critic,
#           optimizers_actors, optimizer_critic, gamma=0.99,
#           max_steps_per_episode=200, device=None, verbose=True,
#           save_dir=None, save_every=10,
#           plot_rewards_fn=None,
#           plot_task_stats=None,
#           plot_values_fn=None,
#           save_models_fn=None,
#           assignment_interval=5):  # Add as parameter

#     episode_rewards = []
#     episode_task_stats = []
#     episode_value_means = []
    
#     if device is None:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     for a in actors.values():
#         a.to(device)
#     critic.to(device)

#     try:
#         for episode in range(num_episodes):
#             # Decay entropy coefficient
#             entropy_coef = max(0.01, 0.05 * (0.995 ** episode))

#             print(f"=== Episode {episode+1}/{num_episodes} ===")

#             obs, _ = env.reset()
#             ego_graphs, attribute_matrix = obs

#             # Buffers for experience collection (only update on decision steps)
#             trajectory_buffer = {
#                 'states': [],
#                 'actions': [],
#                 'rewards': [],
#                 'log_probs': [],
#                 'entropies': [],
#                 'values': [],
#                 'dones': []
#             }

#             episode_values = []
#             done = False
#             episode_reward = 0.0
#             step = 0
#             cumulative_reward = {rid: 0.0 for rid in range(env.n_robots)}

#             while (not done) and step < max_steps_per_episode:
                
#                 # Only collect experience on decision steps
#                 is_decision_step = (step % assignment_interval == 0)
                
#                 if is_decision_step:
#                     x = torch.as_tensor(attribute_matrix, dtype=torch.float32, device=device)
                    
#                     actions = {}
#                     log_probs = {}
#                     entropies = {}
                    
#                     # ================= ACTOR =================
#                     for rid, ego_list in ego_graphs.items():
#                         if len(ego_list) == 0:
#                             actions[rid] = []
#                             continue

#                         edge_index = _build_edge_index_from_ego_list(ego_list).to(device)
                        
#                         # Forward pass
#                         logits = actors[rid](x, edge_index)
                        
#                         nodes_present = torch.unique(edge_index)
#                         task_mask = (nodes_present >= env.n_robots) & \
#                                     (nodes_present < env.n_robots + env.n_tasks)
                        
#                         task_nodes_tensor = nodes_present[task_mask]
                        
#                         if task_nodes_tensor.numel() == 0:
#                             actions[rid] = []
#                             continue
                        
#                         task_logits = logits[task_nodes_tensor]
                        
#                         # Add temperature for exploration
#                         temperature = max(0.5, 1.0 * (0.99 ** episode))
#                         task_logits = task_logits / temperature
                        
#                         dist = torch.distributions.Categorical(logits=task_logits)
#                         sample_idx = dist.sample()
#                         chosen_node = int(task_nodes_tensor[sample_idx].item())
                        
#                         actions[rid] = [chosen_node]
#                         log_probs[rid] = dist.log_prob(sample_idx)
#                         entropies[rid] = dist.entropy()
                    
#                     # ================= CRITIC =================
#                     all_edge_list = []
#                     for ego_list in ego_graphs.values():
#                         if len(ego_list) > 0:
#                             all_edge_list.append(np.concatenate(ego_list, axis=0))
                    
#                     if all_edge_list:
#                         global_edge_index = torch.as_tensor(
#                             np.concatenate(all_edge_list, axis=0),
#                             dtype=torch.long,
#                             device=device
#                         ).t().contiguous()
#                     else:
#                         global_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
                    
#                     critic_out = critic(x, global_edge_index, batch=None, num_robots=env.n_robots)
                    
#                     # Store state for this decision point
#                     trajectory_buffer['states'].append((x.clone(), global_edge_index.clone()))
#                     trajectory_buffer['actions'].append(actions.copy())
#                     trajectory_buffer['log_probs'].append({k: v.clone() for k, v in log_probs.items()})
#                     trajectory_buffer['entropies'].append({k: v.clone() for k, v in entropies.items()})
                    
#                     # Store value predictions
#                     if isinstance(critic_out, torch.Tensor):
#                         trajectory_buffer['values'].append(critic_out.clone())
#                         episode_values.append(critic_out.detach().mean().cpu().item())
#                     else:
#                         trajectory_buffer['values'].append(torch.tensor(critic_out, device=device))
#                         episode_values.append(float(critic_out))
                
#                 else:
#                     actions = None  # No new assignments on non-decision steps
                
#                 # ================= ENV STEP =================
#                 next_obs, reward, done, truncated, info_reward, info = env.step(
#                     actions, assignment_interval=assignment_interval
#                 )
                
#                 # Accumulate rewards between decision steps
#                 if isinstance(reward, dict):
#                     for rid, r in reward.items():
#                         cumulative_reward[rid] += r
#                     episode_reward += sum(reward.values())
#                 else:
#                     episode_reward += reward
#                     cumulative_reward[0] += reward
                
#                 # Store cumulative reward at decision steps
#                 if is_decision_step and len(trajectory_buffer['states']) > 0:
#                     trajectory_buffer['rewards'].append(cumulative_reward.copy())
#                     trajectory_buffer['dones'].append(done or truncated)
#                     # Reset cumulative reward
#                     cumulative_reward = {rid: 0.0 for rid in range(env.n_robots)}
                
#                 ego_graphs, attribute_matrix = next_obs
#                 step += 1
            
#             # ================= LEARNING UPDATE (END OF EPISODE) =================
#             if len(trajectory_buffer['states']) > 0:
                
#                 # Compute returns and advantages
#                 returns = []
#                 advantages = []
                
#                 R = 0  # Bootstrap from terminal state
#                 for t in reversed(range(len(trajectory_buffer['rewards']))):
#                     R = sum(trajectory_buffer['rewards'][t].values()) + gamma * R * (1 - trajectory_buffer['dones'][t])
#                     returns.insert(0, R)
                
#                 # Normalize returns for stability
#                 returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device)
#                 if len(returns_tensor) > 1:
#                     returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)
                
#                 # Compute advantages
#                 for t in range(len(trajectory_buffer['values'])):
#                     value = trajectory_buffer['values'][t]
#                     if isinstance(value, torch.Tensor) and value.numel() > 1:
#                         value = value.mean()
#                     advantages.append(returns_tensor[t] - value)
                
#                 advantages_tensor = torch.stack(advantages)
#                 if len(advantages_tensor) > 1:
#                     advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
                
#                 # ================= CRITIC UPDATE =================
#                 optimizer_critic.zero_grad()
#                 critic_loss = 0
#                 for t in range(len(trajectory_buffer['values'])):
#                     value = trajectory_buffer['values'][t]
#                     if isinstance(value, torch.Tensor) and value.numel() > 1:
#                         value = value.mean()
#                     critic_loss += (returns_tensor[t] - value) ** 2
                
#                 critic_loss = critic_loss / len(trajectory_buffer['values'])
#                 critic_loss.backward()
#                 torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
#                 optimizer_critic.step()
                
#                 # ================= ACTOR UPDATE =================
#                 for rid in range(env.n_robots):
#                     optimizers_actors[rid].zero_grad()
                    
#                     actor_loss = 0
#                     entropy_loss = 0
#                     count = 0
                    
#                     for t in range(len(trajectory_buffer['log_probs'])):
#                         if rid in trajectory_buffer['log_probs'][t]:
#                             log_prob = trajectory_buffer['log_probs'][t][rid]
#                             entropy = trajectory_buffer['entropies'][t].get(rid, torch.tensor(0.0, device=device))
                            
#                             actor_loss += -(log_prob * advantages_tensor[t].detach())
#                             entropy_loss += -entropy
#                             count += 1
                    
#                     if count > 0:
#                         actor_loss = actor_loss / count
#                         entropy_loss = entropy_loss / count
#                         total_loss = actor_loss + entropy_coef * entropy_loss
                        
#                         total_loss.backward()
#                         torch.nn.utils.clip_grad_norm_(actors[rid].parameters(), max_norm=0.5)
#                         optimizers_actors[rid].step()
            
#             # ================= EPISODE TASK STATS =================
#             current_time = env.time_count
#             obsolete = sum(1 for task in env.tasks if task.is_obsolete(current_time))
#             never_picked = sum(1 for task in env.tasks if task.is_obsolete(current_time) and not task.is_pickedup)
#             completed = sum(1 for task in env.tasks if task.is_droppedoff)
            
#             episode_task_stats.append({
#                 "episode": episode + 1,
#                 "obsolete": obsolete,
#                 "never_picked": never_picked,
#                 "completed": completed
#             })
            
#             episode_rewards.append(episode_reward)
            
#             if len(episode_values) > 0:
#                 episode_value_means.append(float(np.mean(episode_values)))
#             else:
#                 episode_value_means.append(0.0)
            
#             print(f"Episode {episode+1}: Reward={episode_reward:.2f}, Completed={completed}, Obsolete={obsolete}, Avg Value={episode_value_means[-1]:.4f}")
            
#             # ================= SAVE =================
#             if save_dir and save_every > 0 and (episode + 1) % save_every == 0:
#                 if save_models_fn:
#                     save_models_fn(save_dir, actors, critic)
#                 if plot_rewards_fn:
#                     plot_rewards_fn(save_dir, episode_rewards)
#                 if plot_task_stats is not None:
#                     plot_task_stats(save_dir, episode_task_stats)
#                 if plot_values_fn is not None:
#                     plot_values_fn(save_dir, episode_value_means)
                
#                 with open(save_dir / "episode_rewards.json", "w") as f:
#                     json.dump([float(x) for x in episode_rewards], f)
#                 with open(save_dir / "episode_task_stats.json", "w") as f:
#                     json.dump(episode_task_stats, f, indent=2)
#                 with open(save_dir / "episode_values.json", "w") as f:
#                     json.dump(episode_value_means, f)

#     except KeyboardInterrupt:
#         print("\nTraining interrupted. Saving progress...")
#         if save_dir and save_models_fn:
#             save_models_fn(save_dir, actors, critic)

#     return episode_rewards, episode_task_stats, episode_value_means
# def train(env, num_episodes, actors, critic,
#           optimizers_actors, optimizer_critic, 
#           schedulers_actors=None,
#           scheduler_critic=None,gamma=0.99,
#           max_steps_per_episode=500, device=None, verbose=True,
#           save_dir=None, save_every=10,
#           plot_rewards_fn=None,
#           plot_task_stats=None,
#           plot_values_fn=None,
#           save_models_fn=None,
#           assignment_interval=5):

#     episode_rewards = []
#     episode_task_stats = []
#     episode_value_means = []
    
#     # 🔥 NEW: Running baseline for value function
#     value_baseline = 0.0
    
#     if device is None:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     for a in actors.values():
#         a.to(device)
#     critic.to(device)

#     try:
#         for episode in range(num_episodes):
#             entropy_coef = max(0.01, 0.05 * (0.995 ** episode))

#             print(f"\n=== Episode {episode+1}/{num_episodes} ===")

#             obs, _ = env.reset()
#             ego_graphs, attribute_matrix = obs

#             trajectory_buffer = {
#                 'states': [],
#                 'actions': [],
#                 'rewards': [],
#                 'log_probs': [],
#                 'entropies': [],
#                 'values': [],
#                 'dones': []
#             }

#             episode_values = []
#             done = False
#             episode_reward = 0.0
#             step = 0
#             cumulative_reward = {rid: 0.0 for rid in range(env.n_robots)}
            
#             meaningful_decisions = 0
#             empty_decisions = 0

#             while (not done) and step < max_steps_per_episode:
                
#                 x = torch.as_tensor(attribute_matrix, dtype=torch.float32, device=device)
                
#                 available_tasks = env.get_available_task_ids()
#                 has_available_robots = any(r.capacity < r.maxCapacity for r in env.robots)
#                 is_meaningful = len(available_tasks) > 0 and has_available_robots
#                 is_decision_step = (step % assignment_interval == 0)
                
#                 if is_decision_step and is_meaningful:
#                     meaningful_decisions += 1
                    
#                     actions = {}
#                     log_probs = {}
#                     entropies = {}
                    
#                     # ================= ACTOR =================
#                     for rid, ego_list in ego_graphs.items():
#                         if len(ego_list) == 0:
#                             actions[rid] = []
#                             continue

#                         edge_index = _build_edge_index_from_ego_list(ego_list).to(device)
#                         logits = actors[rid](x, edge_index)
                        
#                         nodes_present = torch.unique(edge_index)
#                         task_mask = (nodes_present >= env.n_robots) & \
#                                     (nodes_present < env.n_robots + env.n_tasks)
                        
#                         task_nodes_tensor = nodes_present[task_mask]
                        
#                         if task_nodes_tensor.numel() == 0:
#                             actions[rid] = []
#                             continue
                        
#                         task_logits = logits[task_nodes_tensor]
                        
#                         temperature = max(0.5, 1.0 * (0.99 ** episode))
#                         task_logits = task_logits / temperature
                        
#                         dist = torch.distributions.Categorical(logits=task_logits)
#                         sample_idx = dist.sample()
#                         chosen_node = int(task_nodes_tensor[sample_idx].item())
                        
#                         actions[rid] = [chosen_node]
#                         log_probs[rid] = dist.log_prob(sample_idx)
#                         entropies[rid] = dist.entropy()
                    
#                     # ================= CRITIC =================
#                     all_edge_list = []
#                     for ego_list in ego_graphs.values():
#                         if len(ego_list) > 0:
#                             all_edge_list.append(np.concatenate(ego_list, axis=0))
                    
#                     if all_edge_list:
#                         global_edge_index = torch.as_tensor(
#                             np.concatenate(all_edge_list, axis=0),
#                             dtype=torch.long,
#                             device=device
#                         ).t().contiguous()
#                     else:
#                         global_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
                    
#                     critic_out = critic(x, global_edge_index, batch=None, num_robots=env.n_robots)
                    
#                     # Store trajectory
#                     trajectory_buffer['states'].append((x.clone(), global_edge_index.clone()))
#                     trajectory_buffer['actions'].append(actions.copy())
#                     trajectory_buffer['log_probs'].append({k: v.clone() for k, v in log_probs.items()})
#                     trajectory_buffer['entropies'].append({k: v.clone() for k, v in entropies.items()})
                    
#                     if isinstance(critic_out, torch.Tensor):
#                         trajectory_buffer['values'].append(critic_out.clone())
#                         episode_values.append(critic_out.detach().mean().cpu().item())
#                     else:
#                         trajectory_buffer['values'].append(torch.tensor(critic_out, device=device))
#                         episode_values.append(float(critic_out))
                    
#                     trajectory_buffer['rewards'].append(cumulative_reward.copy())
#                     trajectory_buffer['dones'].append(False)
                    
#                     cumulative_reward = {rid: 0.0 for rid in range(env.n_robots)}
                    
#                 elif is_decision_step:
#                     empty_decisions += 1
#                     actions = None
#                 else:
#                     actions = None
                
#                 # ================= ENV STEP =================
#                 next_obs, reward, done, truncated, info_reward, info = env.step(
#                     actions, assignment_interval=assignment_interval
#                 )
                
#                 if isinstance(reward, dict):
#                     for rid, r in reward.items():
#                         cumulative_reward[rid] += r
#                     episode_reward += sum(reward.values())
#                 else:
#                     episode_reward += reward
#                     cumulative_reward[0] += reward
                
#                 ego_graphs, attribute_matrix = next_obs
#                 step += 1
            
#             if len(trajectory_buffer['dones']) > 0:
#                 trajectory_buffer['dones'][-1] = True
            
#             # ================= LEARNING UPDATE =================
#             if len(trajectory_buffer['states']) > 0:
                
#                 # 🔥 FIXED: Compute returns WITHOUT aggressive normalization
#                 returns = []
#                 R = 0
                
#                 for t in reversed(range(len(trajectory_buffer['rewards']))):
#                     reward_sum = sum(trajectory_buffer['rewards'][t].values())
#                     R = reward_sum + gamma * R * (1 - trajectory_buffer['dones'][t])
#                     returns.insert(0, R)
                
#                 returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device)
                
#                 # 🔥 FIXED: Use running baseline instead of per-episode normalization
#                 # Update running baseline (exponential moving average)
#                 current_return_mean = returns_tensor.mean().item()
#                 value_baseline = 0.95 * value_baseline + 0.05 * current_return_mean
                
#                 # 🔥 FIXED: Only subtract baseline, NO normalization
#                 # This preserves the scale of returns
#                 returns_normalized = returns_tensor - value_baseline
                
#                 # 🔥 FIXED: Clip instead of normalize to prevent extreme values
#                 returns_normalized = torch.clamp(returns_normalized, -100, 100)
                
#                 # Compute advantages
#                 advantages = []
#                 for t in range(len(trajectory_buffer['values'])):
#                     value = trajectory_buffer['values'][t]
#                     if isinstance(value, torch.Tensor) and value.numel() > 1:
#                         value = value.mean()
#                     advantages.append(returns_normalized[t] - value)
                
#                 advantages_tensor = torch.stack(advantages)
                
#                 # 🔥 FIXED: Light normalization for advantages only (not returns)
#                 # This helps with actor learning but doesn't destroy value function
#                 if len(advantages_tensor) > 1 and advantages_tensor.std() > 1e-3:
#                     advantages_tensor = advantages_tensor / (advantages_tensor.std() + 1e-3)
#                 advantages_tensor = torch.clamp(advantages_tensor, -10, 10)
                
#                 # ================= CRITIC UPDATE =================
#                 optimizer_critic.zero_grad()
#                 critic_loss = 0
#                 for t in range(len(trajectory_buffer['values'])):
#                     value = trajectory_buffer['values'][t]
#                     if isinstance(value, torch.Tensor) and value.numel() > 1:
#                         value = value.mean()
#                     # 🔥 FIXED: Train against returns_normalized, not fully normalized
#                     critic_loss += (returns_normalized[t] - value) ** 2
                
#                 critic_loss = critic_loss / len(trajectory_buffer['values'])
                
#                 # 🔥 NEW: Add value function regularization to prevent collapse
#                 if len(trajectory_buffer['values']) > 1:
#                     values_stacked = torch.stack([
#                         v.mean() if (isinstance(v, torch.Tensor) and v.numel() > 1) else v
#                         for v in trajectory_buffer['values']
#                     ])
#                     value_variance = values_stacked.var()
#                     # Penalize if variance gets too small (prevents collapse to constant)
#                     variance_penalty = torch.relu(0.1 - value_variance) * 10.0
#                     critic_loss = critic_loss + variance_penalty
                
#                 critic_loss.backward()
#                 torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
#                 optimizer_critic.step()
                
#                 # ================= ACTOR UPDATE =================
#                 for rid in range(env.n_robots):
#                     optimizers_actors[rid].zero_grad()
                    
#                     actor_loss = 0
#                     entropy_loss = 0
#                     count = 0
                    
#                     for t in range(len(trajectory_buffer['log_probs'])):
#                         if rid in trajectory_buffer['log_probs'][t]:
#                             log_prob = trajectory_buffer['log_probs'][t][rid]
#                             entropy = trajectory_buffer['entropies'][t].get(rid, torch.tensor(0.0, device=device))
                            
#                             actor_loss += -(log_prob * advantages_tensor[t].detach())
#                             entropy_loss += -entropy
#                             count += 1
                    
#                     if count > 0:
#                         actor_loss = actor_loss / count
#                         entropy_loss = entropy_loss / count
#                         total_loss = actor_loss + entropy_coef * entropy_loss
                        
#                         total_loss.backward()
#                         torch.nn.utils.clip_grad_norm_(actors[rid].parameters(), max_norm=0.5)
#                         optimizers_actors[rid].step()
            
#             # ================= EPISODE STATS =================
#             current_time = env.time_count
#             obsolete = sum(1 for task in env.tasks if task.is_obsolete(current_time))
#             never_picked = sum(1 for task in env.tasks if task.is_obsolete(current_time) and not task.is_pickedup)
#             completed = sum(1 for task in env.tasks if task.is_droppedoff)
            
#             episode_task_stats.append({
#                 "episode": episode + 1,
#                 "obsolete": obsolete,
#                 "never_picked": never_picked,
#                 "completed": completed
#             })
            
#             episode_rewards.append(episode_reward)
            
#             if len(episode_values) > 0:
#                 episode_value_means.append(float(np.mean(episode_values)))
#             else:
#                 episode_value_means.append(0.0)
            
#             print(f"Episode {episode+1}: Reward={episode_reward:.2f}, Completed={completed}/{env.n_tasks}, "
#                   f"Obsolete={obsolete}, Meaningful={meaningful_decisions}, Empty={empty_decisions}, "
#                   f"Avg Value={episode_value_means[-1]:.4f}, Baseline={value_baseline:.4f}")
            
#             # ================= SAVE =================
#             if save_dir and save_every > 0 and (episode + 1) % save_every == 0:
#                 if save_models_fn:
#                     save_models_fn(save_dir, actors, critic)
#                 if plot_rewards_fn:
#                     plot_rewards_fn(save_dir, episode_rewards)
#                 if plot_task_stats is not None:
#                     plot_task_stats(save_dir, episode_task_stats)
#                 if plot_values_fn is not None:
#                     plot_values_fn(save_dir, episode_value_means)
                
#                 with open(save_dir / "episode_rewards.json", "w") as f:
#                     json.dump([float(x) for x in episode_rewards], f)
#                 with open(save_dir / "episode_task_stats.json", "w") as f:
#                     json.dump(episode_task_stats, f, indent=2)
#                 with open(save_dir / "episode_values.json", "w") as f:
#                     json.dump(episode_value_means, f)

#     except KeyboardInterrupt:
#         print("\nTraining interrupted. Saving progress...")
#         if save_dir and save_models_fn:
#             save_models_fn(save_dir, actors, critic)

#     return episode_rewards, episode_task_stats, episode_value_means

def train(env, num_episodes, actors, critic,
          optimizers_actors, optimizer_critic, 
          schedulers_actors=None,
          scheduler_critic=None,
          gamma=0.99,
          max_steps_per_episode=500, device=None, verbose=True,
          save_dir=None, save_every=10,
          plot_rewards_fn=None,
          plot_task_stats=None,
          plot_values_fn=None,
          save_models_fn=None,
          assignment_interval=5,
          n_step=20):  # 🔥 NEW PARAMETER

    episode_rewards = []
    episode_task_stats = []
    episode_value_means = []
    critic_losses = [] 
    actor_losses = []
    
    value_baseline = None
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for a in actors.values():
        a.to(device)
    critic.to(device)
    
    import copy
    critic_target = copy.deepcopy(critic)
    critic_target.eval()
    for param in critic_target.parameters():
        param.requires_grad = False

    # 🔥 NEW: n-step return computation function
    def compute_n_step_returns(rewards_list, values_list, dones_list, gamma, n_steps):
        """
        Compute n-step TD returns for better credit assignment.
        
        Args:
            rewards_list: List of reward dicts per decision step
            values_list: List of value predictions per decision step
            dones_list: List of done flags per decision step
            gamma: Discount factor
            n_steps: Number of steps to look ahead
        
        Returns:
            List of n-step returns (one per decision step)
        """
        returns = []
        T = len(rewards_list)
        
        for t in range(T):
            n_step_return = 0.0
            
            # Accumulate rewards for the next n steps
            for k in range(n_steps):
                if t + k >= T:
                    break
                
                # Sum rewards across all robots at this timestep
                reward_sum = sum(rewards_list[t + k].values())
                
                # Add discounted reward
                n_step_return += (gamma ** k) * reward_sum
                
                # Stop if episode ended
                if dones_list[t + k]:
                    break
            
            # Bootstrap with value function if episode continues beyond n steps
            if t + n_steps < T and not dones_list[min(t + n_steps - 1, T - 1)]:
                bootstrap_value = values_list[t + n_steps]
                
                # Handle tensor vs scalar value
                if isinstance(bootstrap_value, torch.Tensor):
                    bootstrap_value = bootstrap_value.mean().detach().item()
                else:
                    bootstrap_value = float(bootstrap_value)
                
                n_step_return += (gamma ** n_steps) * bootstrap_value
            
            returns.append(n_step_return)
        
        return returns

    try:
        for episode in range(num_episodes):
            entropy_coef = max(0.1, 0.3 * (0.998 ** episode))

            if episode % 10 == 0 or episode < 20:
                print(f"\n=== Episode {episode+1}/{num_episodes} ===")

            obs, _ = env.reset()
            ego_graphs, attribute_matrix = obs

            trajectory_buffer = {
                'states': [],
                'actions': [],
                'rewards': [],
                'log_probs': [],
                'entropies': [],
                'values': [],
                'dones': [],
                'next_states': []
            }

            episode_values = []
            done = False
            episode_reward = 0.0
            step = 0
            cumulative_reward = {rid: 0.0 for rid in range(env.n_robots)}
            
            meaningful_decisions = 0
            empty_decisions = 0

            while (not done) and step < max_steps_per_episode:
                
                x = torch.as_tensor(attribute_matrix, dtype=torch.float32, device=device)
                
                available_tasks = env.get_available_task_ids()
                has_available_robots = any(r.capacity < r.maxCapacity for r in env.robots)
                is_meaningful = len(available_tasks) > 0 and has_available_robots
                is_decision_step = (step % assignment_interval == 0)
                
                if is_decision_step and is_meaningful:
                    meaningful_decisions += 1
                    
                    actions = {}
                    log_probs = {}
                    entropies = {}
                    
                    for rid, ego_list in ego_graphs.items():
                        if len(ego_list) == 0:
                            actions[rid] = []
                            continue

                        edge_index = _build_edge_index_from_ego_list(ego_list).to(device)
                        logits = actors[rid](x, edge_index)
                        
                        nodes_present = torch.unique(edge_index)
                        task_mask = (nodes_present >= env.n_robots) & \
                                    (nodes_present < env.n_robots + env.n_tasks)
                        
                        task_nodes_tensor = nodes_present[task_mask]
                        
                        if task_nodes_tensor.numel() == 0:
                            actions[rid] = []
                            continue
                        
                        task_logits = logits[task_nodes_tensor]
                        temperature = max(0.7, 1.2 * (0.995 ** episode))
                        task_logits = task_logits / temperature
                        
                        dist = torch.distributions.Categorical(logits=task_logits)
                        sample_idx = dist.sample()
                        chosen_node = int(task_nodes_tensor[sample_idx].item())
                        
                        actions[rid] = [chosen_node]
                        log_probs[rid] = dist.log_prob(sample_idx)
                        entropies[rid] = dist.entropy()
                    
                    all_edge_list = []
                    for ego_list in ego_graphs.values():
                        if len(ego_list) > 0:
                            all_edge_list.append(np.concatenate(ego_list, axis=0))
                    
                    if all_edge_list:
                        global_edge_index = torch.as_tensor(
                            np.concatenate(all_edge_list, axis=0),
                            dtype=torch.long,
                            device=device
                        ).t().contiguous()
                    else:
                        global_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
                    
                    critic_out = critic(x, global_edge_index, batch=None, num_robots=env.n_robots)
                    
                    trajectory_buffer['states'].append((x.clone(), global_edge_index.clone()))
                    trajectory_buffer['actions'].append(actions.copy())
                    trajectory_buffer['log_probs'].append({k: v.clone() for k, v in log_probs.items()})
                    trajectory_buffer['entropies'].append({k: v.clone() for k, v in entropies.items()})
                    
                    if isinstance(critic_out, torch.Tensor):
                        trajectory_buffer['values'].append(critic_out.clone())
                        episode_values.append(critic_out.detach().mean().cpu().item())
                    else:
                        trajectory_buffer['values'].append(torch.tensor(critic_out, device=device))
                        episode_values.append(float(critic_out))
                    
                    trajectory_buffer['rewards'].append(cumulative_reward.copy())
                    trajectory_buffer['dones'].append(False)
                    
                    cumulative_reward = {rid: 0.0 for rid in range(env.n_robots)}
                    
                elif is_decision_step:
                    empty_decisions += 1
                    actions = None
                else:
                    actions = None
                
                next_obs, reward, done, truncated, info_reward, info = env.step(
                    actions, assignment_interval=assignment_interval
                )
                
                if isinstance(reward, dict):
                    for rid, r in reward.items():
                        cumulative_reward[rid] += r
                    episode_reward += sum(reward.values())
                else:
                    episode_reward += reward
                    cumulative_reward[0] += reward
                
                ego_graphs_next, attribute_matrix_next = next_obs
                
                if is_decision_step and is_meaningful and len(trajectory_buffer['states']) > 0:
                    x_next = torch.as_tensor(attribute_matrix_next, dtype=torch.float32, device=device)
                    
                    all_edge_list_next = []
                    for ego_list in ego_graphs_next.values():
                        if len(ego_list) > 0:
                            all_edge_list_next.append(np.concatenate(ego_list, axis=0))
                    
                    if all_edge_list_next:
                        global_edge_index_next = torch.as_tensor(
                            np.concatenate(all_edge_list_next, axis=0),
                            dtype=torch.long,
                            device=device
                        ).t().contiguous()
                    else:
                        global_edge_index_next = torch.empty((2, 0), dtype=torch.long, device=device)
                    
                    trajectory_buffer['next_states'].append((x_next.clone(), global_edge_index_next.clone()))
                
                ego_graphs = ego_graphs_next
                attribute_matrix = attribute_matrix_next
                step += 1
            
            if len(trajectory_buffer['dones']) > 0:
                trajectory_buffer['dones'][-1] = True
            
            # ================= LEARNING UPDATE =================
            if len(trajectory_buffer['states']) > 0:
                
                # 🔥 NEW: Compute n-step returns instead of Monte Carlo
                returns = compute_n_step_returns(
                    trajectory_buffer['rewards'],
                    trajectory_buffer['values'],
                    trajectory_buffer['dones'],
                    gamma=gamma,
                    n_steps=n_step
                )
                
                returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device)
                
                # Update baseline for logging (not used in training)
                if value_baseline is None:
                    value_baseline = episode_reward
                    print(f"Initialized baseline to {value_baseline:.2f}")
                elif episode < 5:
                    value_baseline = 0.3 * value_baseline + 0.7 * episode_reward
                    print(f"Episode {episode+1}: Fast baseline adaptation: {value_baseline:.2f} (reward was {episode_reward:.2f})")
                elif episode < 20:
                    value_baseline = 0.7 * value_baseline + 0.3 * episode_reward
                    if episode % 5 == 0 or episode < 7:
                        print(f"Episode {episode+1}: Moderate baseline adaptation: {value_baseline:.2f} (reward was {episode_reward:.2f})")
                elif episode < 100:
                    value_baseline = 0.9 * value_baseline + 0.1 * episode_reward
                else:
                    value_baseline = 0.99 * value_baseline + 0.01 * episode_reward
                
                # Clip extreme values (but don't subtract baseline)
                returns_normalized = torch.clamp(returns_tensor, -10000, 10000)
                
                # Compute advantages (for actor)
                advantages = []
                for t in range(len(trajectory_buffer['values'])):
                    value = trajectory_buffer['values'][t]
                    if isinstance(value, torch.Tensor) and value.numel() > 1:
                        value = value.mean()
                    advantages.append(returns_normalized[t] - value)
                
                advantages_tensor = torch.stack(advantages)
                
                # Normalize advantages ONLY (not returns)
                if len(advantages_tensor) > 1:
                    adv_std = advantages_tensor.std()
                    if adv_std > 1.0:
                        advantages_tensor = advantages_tensor / (adv_std + 1e-3)
                advantages_tensor = torch.clamp(advantages_tensor, -10, 10)
                
                # ================= CRITIC UPDATE =================
                optimizer_critic.zero_grad()
                critic_loss = 0
                for t in range(len(trajectory_buffer['values'])):
                    value = trajectory_buffer['values'][t]
                    if isinstance(value, torch.Tensor) and value.numel() > 1:
                        value = value.mean()
                    # Train critic to predict n-step returns
                    critic_loss += (returns_normalized[t] - value) ** 2
                
                critic_loss = critic_loss / len(trajectory_buffer['values'])
                
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
                optimizer_critic.step()
                critic_losses.append(critic_loss.item())
                
                # Soft update target network
                tau = 0.005
                for target_param, param in zip(critic_target.parameters(), critic.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                
                if scheduler_critic is not None:
                    scheduler_critic.step()
                
                # ================= ACTOR UPDATE =================
                total_actor_loss = 0
                for rid in range(env.n_robots):
                    optimizers_actors[rid].zero_grad()
                    
                    actor_loss = 0
                    entropy_loss = 0
                    count = 0
                    
                    for t in range(len(trajectory_buffer['log_probs'])):
                        if rid in trajectory_buffer['log_probs'][t]:
                            log_prob = trajectory_buffer['log_probs'][t][rid]
                            entropy = trajectory_buffer['entropies'][t].get(rid, torch.tensor(0.0, device=device))
                            
                            actor_loss += -(log_prob * advantages_tensor[t].detach())
                            entropy_loss += -entropy
                            count += 1
                    
                    if count > 0:
                        actor_loss = actor_loss / count
                        entropy_loss = entropy_loss / count
                        total_loss = actor_loss + entropy_coef * entropy_loss
                        
                        total_loss.backward()
                        torch.nn.utils.clip_grad_norm_(actors[rid].parameters(), max_norm=0.5)
                        optimizers_actors[rid].step()
                        total_actor_loss += total_loss.item()
                        
                        if schedulers_actors is not None and rid in schedulers_actors:
                            schedulers_actors[rid].step()
                actor_losses.append(total_actor_loss)
            
            # ================= EPISODE STATS =================
            current_time = env.time_count
            obsolete = sum(1 for task in env.tasks if task.is_obsolete(current_time))
            never_picked = sum(1 for task in env.tasks if task.is_obsolete(current_time) and not task.is_pickedup)
            completed = sum(1 for task in env.tasks if task.is_droppedoff)
            
            episode_task_stats.append({
                "episode": episode + 1,
                "obsolete": obsolete,
                "never_picked": never_picked,
                "completed": completed
            })
            
            episode_rewards.append(episode_reward)
            
            if len(episode_values) > 0:
                episode_value_means.append(float(np.mean(episode_values)))
            else:
                episode_value_means.append(0.0)
            
            current_critic_lr = optimizer_critic.param_groups[0]['lr']
            current_actor_lr = optimizers_actors[0].param_groups[0]['lr']
            
            if episode % 10 == 0 or episode < 20:
                print(f"Episode {episode+1}: Reward={episode_reward:.2f}, Completed={completed}/{env.n_tasks}, "
                      f"Obsolete={obsolete}, Meaningful={meaningful_decisions}, "
                      f"Value={episode_value_means[-1]:.2f}, Baseline={value_baseline:.2f}, "
                      f"CriticLoss={(critic_loss.item() if 'critic_loss' in locals() else 0):.4f}, "
                      f"LR_c={current_critic_lr:.2e}, LR_a={current_actor_lr:.2e}, "
                      f"n_step={n_step}")  # 🔥 NEW: Show n_step value
            
            # ================= SAVE =================
            if save_dir and save_every > 0 and (episode + 1) % save_every == 0:
                if save_models_fn:
                    save_models_fn(save_dir, actors, critic)
                if plot_rewards_fn:
                    plot_rewards_fn(save_dir, episode_rewards)
                if plot_task_stats is not None:
                    plot_task_stats(save_dir, episode_task_stats)
                if plot_values_fn is not None:
                    plot_values_fn(save_dir, episode_value_means)
                
                with open(save_dir / "episode_rewards.json", "w") as f:
                    json.dump([float(x) for x in episode_rewards], f)
                with open(save_dir / "episode_task_stats.json", "w") as f:
                    json.dump(episode_task_stats, f, indent=2)
                with open(save_dir / "episode_values.json", "w") as f:
                    json.dump(episode_value_means, f)

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving progress...")
        if save_dir and save_models_fn:
            save_models_fn(save_dir, actors, critic)

    return episode_rewards, episode_task_stats, episode_value_means, critic_losses, actor_losses
# def train(env, num_episodes, actors, critic,
#           optimizers_actors, optimizer_critic, 
#           schedulers_actors=None,
#           scheduler_critic=None,
#           gamma=0.99,
#           max_steps_per_episode=500, device=None, verbose=True,
#           save_dir=None, save_every=10,
#           plot_rewards_fn=None,
#           plot_task_stats=None,
#           plot_values_fn=None,
#           save_models_fn=None,
#           assignment_interval=5):

#     episode_rewards = []
#     episode_task_stats = []
#     episode_value_means = []
#     critic_losses = [] 
#     actor_losses = []
    
#     value_baseline = None  # ← Initialize to None
    
#     if device is None:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     for a in actors.values():
#         a.to(device)
#     critic.to(device)
    
#     import copy
#     critic_target = copy.deepcopy(critic)
#     critic_target.eval()
#     for param in critic_target.parameters():
#         param.requires_grad = False

#     try:
#         for episode in range(num_episodes):
#             entropy_coef = max(0.1, 0.3 * (0.998 ** episode))

#             if episode % 10 == 0 or episode < 20:
#                 print(f"\n=== Episode {episode+1}/{num_episodes} ===")

#             obs, _ = env.reset()
#             ego_graphs, attribute_matrix = obs

#             trajectory_buffer = {
#                 'states': [],
#                 'actions': [],
#                 'rewards': [],
#                 'log_probs': [],
#                 'entropies': [],
#                 'values': [],
#                 'dones': [],
#                 'next_states': []
#             }

#             episode_values = []
#             done = False
#             episode_reward = 0.0
#             step = 0
#             cumulative_reward = {rid: 0.0 for rid in range(env.n_robots)}
            
#             meaningful_decisions = 0
#             empty_decisions = 0

#             while (not done) and step < max_steps_per_episode:
                
#                 x = torch.as_tensor(attribute_matrix, dtype=torch.float32, device=device)
                
#                 available_tasks = env.get_available_task_ids()
#                 has_available_robots = any(r.capacity < r.maxCapacity for r in env.robots)
#                 is_meaningful = len(available_tasks) > 0 and has_available_robots
#                 is_decision_step = (step % assignment_interval == 0)
                
#                 if is_decision_step and is_meaningful:
#                     meaningful_decisions += 1
                    
#                     actions = {}
#                     log_probs = {}
#                     entropies = {}
                    
#                     for rid, ego_list in ego_graphs.items():
#                         if len(ego_list) == 0:
#                             actions[rid] = []
#                             continue

#                         edge_index = _build_edge_index_from_ego_list(ego_list).to(device)
#                         logits = actors[rid](x, edge_index)
                        
#                         nodes_present = torch.unique(edge_index)
#                         task_mask = (nodes_present >= env.n_robots) & \
#                                     (nodes_present < env.n_robots + env.n_tasks)
                        
#                         task_nodes_tensor = nodes_present[task_mask]
                        
#                         if task_nodes_tensor.numel() == 0:
#                             actions[rid] = []
#                             continue
                        
#                         task_logits = logits[task_nodes_tensor]
#                         temperature = max(0.7, 1.2 * (0.995 ** episode))
#                         task_logits = task_logits / temperature
                        
#                         dist = torch.distributions.Categorical(logits=task_logits)
#                         sample_idx = dist.sample()
#                         chosen_node = int(task_nodes_tensor[sample_idx].item())
                        
#                         actions[rid] = [chosen_node]
#                         log_probs[rid] = dist.log_prob(sample_idx)
#                         entropies[rid] = dist.entropy()
                    
#                     all_edge_list = []
#                     for ego_list in ego_graphs.values():
#                         if len(ego_list) > 0:
#                             all_edge_list.append(np.concatenate(ego_list, axis=0))
                    
#                     if all_edge_list:
#                         global_edge_index = torch.as_tensor(
#                             np.concatenate(all_edge_list, axis=0),
#                             dtype=torch.long,
#                             device=device
#                         ).t().contiguous()
#                     else:
#                         global_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
                    
#                     critic_out = critic(x, global_edge_index, batch=None, num_robots=env.n_robots)
                    
#                     trajectory_buffer['states'].append((x.clone(), global_edge_index.clone()))
#                     trajectory_buffer['actions'].append(actions.copy())
#                     trajectory_buffer['log_probs'].append({k: v.clone() for k, v in log_probs.items()})
#                     trajectory_buffer['entropies'].append({k: v.clone() for k, v in entropies.items()})
                    
#                     if isinstance(critic_out, torch.Tensor):
#                         trajectory_buffer['values'].append(critic_out.clone())
#                         episode_values.append(critic_out.detach().mean().cpu().item())
#                     else:
#                         trajectory_buffer['values'].append(torch.tensor(critic_out, device=device))
#                         episode_values.append(float(critic_out))
                    
#                     trajectory_buffer['rewards'].append(cumulative_reward.copy())
#                     trajectory_buffer['dones'].append(False)
                    
#                     cumulative_reward = {rid: 0.0 for rid in range(env.n_robots)}
                    
#                 elif is_decision_step:
#                     empty_decisions += 1
#                     actions = None
#                 else:
#                     actions = None
                
#                 next_obs, reward, done, truncated, info_reward, info = env.step(
#                     actions, assignment_interval=assignment_interval
#                 )
                
#                 if isinstance(reward, dict):
#                     for rid, r in reward.items():
#                         cumulative_reward[rid] += r
#                     episode_reward += sum(reward.values())
#                 else:
#                     episode_reward += reward
#                     cumulative_reward[0] += reward
                
#                 ego_graphs_next, attribute_matrix_next = next_obs
                
#                 if is_decision_step and is_meaningful and len(trajectory_buffer['states']) > 0:
#                     x_next = torch.as_tensor(attribute_matrix_next, dtype=torch.float32, device=device)
                    
#                     all_edge_list_next = []
#                     for ego_list in ego_graphs_next.values():
#                         if len(ego_list) > 0:
#                             all_edge_list_next.append(np.concatenate(ego_list, axis=0))
                    
#                     if all_edge_list_next:
#                         global_edge_index_next = torch.as_tensor(
#                             np.concatenate(all_edge_list_next, axis=0),
#                             dtype=torch.long,
#                             device=device
#                         ).t().contiguous()
#                     else:
#                         global_edge_index_next = torch.empty((2, 0), dtype=torch.long, device=device)
                    
#                     trajectory_buffer['next_states'].append((x_next.clone(), global_edge_index_next.clone()))
                
#                 ego_graphs = ego_graphs_next
#                 attribute_matrix = attribute_matrix_next
#                 step += 1
            
#             if len(trajectory_buffer['dones']) > 0:
#                 trajectory_buffer['dones'][-1] = True
            
#             # ================= LEARNING UPDATE =================
#             if len(trajectory_buffer['states']) > 0:
    
#                 returns = []
#                 td_targets = []
                
#                 for t in range(len(trajectory_buffer['rewards'])):
#                     reward_sum = sum(trajectory_buffer['rewards'][t].values())
                    
#                     if trajectory_buffer['dones'][t]:
#                         bootstrap_value = 0.0
#                     elif t < len(trajectory_buffer['next_states']):
#                         x_next, edge_next = trajectory_buffer['next_states'][t]
#                         with torch.no_grad():
#                             next_value = critic_target(x_next, edge_next, batch=None, num_robots=env.n_robots)
#                             if isinstance(next_value, torch.Tensor):
#                                 bootstrap_value = next_value.mean().item()
#                             else:
#                                 bootstrap_value = float(next_value)
#                     else:
#                         bootstrap_value = 0.0
                    
#                     td_target = reward_sum + gamma * bootstrap_value
#                     td_targets.append(td_target)
                
#                 R = 0
#                 for t in reversed(range(len(trajectory_buffer['rewards']))):
#                     reward_sum = sum(trajectory_buffer['rewards'][t].values())
#                     R = reward_sum + gamma * R * (1 - trajectory_buffer['dones'][t])
#                     returns.insert(0, R)
                
#                 returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device)
#                 td_targets_tensor = torch.tensor(td_targets, dtype=torch.float32, device=device)
                
#                 lambda_blend = 0.95
#                 blended_targets = lambda_blend * returns_tensor + (1 - lambda_blend) * td_targets_tensor
                
#                 # 🔥 FIXED: Track baseline for LOGGING only (not for critic training)
#                 if value_baseline is None:
#                     value_baseline = episode_reward
#                     print(f"Initialized baseline to {value_baseline:.2f}")
#                 elif episode < 5:
#                     value_baseline = 0.3 * value_baseline + 0.7 * episode_reward
#                     print(f"Episode {episode+1}: Fast baseline adaptation: {value_baseline:.2f} (reward was {episode_reward:.2f})")
#                 elif episode < 20:
#                     value_baseline = 0.7 * value_baseline + 0.3 * episode_reward
#                     if episode % 5 == 0 or episode < 7:
#                         print(f"Episode {episode+1}: Moderate baseline adaptation: {value_baseline:.2f} (reward was {episode_reward:.2f})")
#                 elif episode < 100:
#                     value_baseline = 0.9 * value_baseline + 0.1 * episode_reward
#                 else:
#                     value_baseline = 0.99 * value_baseline + 0.01 * episode_reward
                
#                 # 🔥 NEW: DON'T subtract baseline - use raw returns for critic
#                 # Just clip extreme values
#                 returns_normalized = torch.clamp(blended_targets, -10000, 10000)
                
#                 # Compute advantages (for actor)
#                 advantages = []
#                 for t in range(len(trajectory_buffer['values'])):
#                     value = trajectory_buffer['values'][t]
#                     if isinstance(value, torch.Tensor) and value.numel() > 1:
#                         value = value.mean()
#                     advantages.append(returns_normalized[t] - value)
                
#                 advantages_tensor = torch.stack(advantages)
                
#                 # Normalize advantages ONLY (not returns)
#                 if len(advantages_tensor) > 1:
#                     adv_std = advantages_tensor.std()
#                     if adv_std > 1.0:
#                         advantages_tensor = advantages_tensor / (adv_std + 1e-3)
#                 advantages_tensor = torch.clamp(advantages_tensor, -10, 10)
                
#                 # ================= CRITIC UPDATE =================
#                 optimizer_critic.zero_grad()
#                 critic_loss = 0
#                 for t in range(len(trajectory_buffer['values'])):
#                     value = trajectory_buffer['values'][t]
#                     if isinstance(value, torch.Tensor) and value.numel() > 1:
#                         value = value.mean()
#                     # 🔥 FIXED: Use RAW returns, not baseline-subtracted
#                     critic_loss += (returns_normalized[t] - value) ** 2
                
#                 critic_loss = critic_loss / len(trajectory_buffer['values'])
                
#                 critic_loss.backward()
#                 torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
#                 optimizer_critic.step()
#                 critic_losses.append(critic_loss.item())
                
#                 # Soft update target network
#                 tau = 0.005
#                 for target_param, param in zip(critic_target.parameters(), critic.parameters()):
#                     target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                
#                 if scheduler_critic is not None:
#                     scheduler_critic.step()
                
#                 # ================= ACTOR UPDATE (unchanged) =================
#                 total_actor_loss = 0
#                 for rid in range(env.n_robots):
#                     optimizers_actors[rid].zero_grad()
                    
#                     actor_loss = 0
#                     entropy_loss = 0
#                     count = 0
                    
#                     for t in range(len(trajectory_buffer['log_probs'])):
#                         if rid in trajectory_buffer['log_probs'][t]:
#                             log_prob = trajectory_buffer['log_probs'][t][rid]
#                             entropy = trajectory_buffer['entropies'][t].get(rid, torch.tensor(0.0, device=device))
                            
#                             actor_loss += -(log_prob * advantages_tensor[t].detach())
#                             entropy_loss += -entropy
#                             count += 1
                    
#                     if count > 0:
#                         actor_loss = actor_loss / count
#                         entropy_loss = entropy_loss / count
#                         total_loss = actor_loss + entropy_coef * entropy_loss
                        
#                         total_loss.backward()
#                         torch.nn.utils.clip_grad_norm_(actors[rid].parameters(), max_norm=0.5)
#                         optimizers_actors[rid].step()
#                         total_actor_loss += total_loss.item()
                        
#                         if schedulers_actors is not None and rid in schedulers_actors:
#                             schedulers_actors[rid].step()
#                 actor_losses.append(total_actor_loss)
            
#             # ================= EPISODE STATS =================
#             current_time = env.time_count
#             obsolete = sum(1 for task in env.tasks if task.is_obsolete(current_time))
#             never_picked = sum(1 for task in env.tasks if task.is_obsolete(current_time) and not task.is_pickedup)
#             completed = sum(1 for task in env.tasks if task.is_droppedoff)
            
#             episode_task_stats.append({
#                 "episode": episode + 1,
#                 "obsolete": obsolete,
#                 "never_picked": never_picked,
#                 "completed": completed
#             })
            
#             episode_rewards.append(episode_reward)
            
#             if len(episode_values) > 0:
#                 episode_value_means.append(float(np.mean(episode_values)))
#             else:
#                 episode_value_means.append(0.0)
            
#             current_critic_lr = optimizer_critic.param_groups[0]['lr']
#             current_actor_lr = optimizers_actors[0].param_groups[0]['lr']
            
#             if episode % 10 == 0 or episode < 20:
#                 print(f"Episode {episode+1}: Reward={episode_reward:.2f}, Completed={completed}/{env.n_tasks}, "
#                       f"Obsolete={obsolete}, Meaningful={meaningful_decisions}, "
#                       f"Value={episode_value_means[-1]:.2f}, Baseline={value_baseline:.2f}, "
#                       f"CriticLoss={(critic_loss.item() if 'critic_loss' in locals() else 0):.4f}, "
#                       f"LR_c={current_critic_lr:.2e}, LR_a={current_actor_lr:.2e}")
                
#                 if isinstance(info_reward, dict):
#                     sparsity = info_reward.get('sum_rewards', 0)
#                     print(f"Episode {episode+1}: Reward={episode_reward:.2f}, Completed={completed}/{env.n_robots * 20}, "
#                           f"Pickups={info_reward.get('pickups_this_step', 0)}, "
#                           f"Deliveries={info_reward.get('deliveries_this_step', 0)}, "
#                           f"Obsolete={info_reward.get('obsolete_this_step', 0)}, "
#                           f"RewardSparsity={info_reward.get('sum_rewards', 0)}%")
            
#             # ================= SAVE =================
#             if save_dir and save_every > 0 and (episode + 1) % save_every == 0:
#                 if save_models_fn:
#                     save_models_fn(save_dir, actors, critic)
#                 if plot_rewards_fn:
#                     plot_rewards_fn(save_dir, episode_rewards)
#                 if plot_task_stats is not None:
#                     plot_task_stats(save_dir, episode_task_stats)
#                 if plot_values_fn is not None:
#                     plot_values_fn(save_dir, episode_value_means)
                
#                 with open(save_dir / "episode_rewards.json", "w") as f:
#                     json.dump([float(x) for x in episode_rewards], f)
#                 with open(save_dir / "episode_task_stats.json", "w") as f:
#                     json.dump(episode_task_stats, f, indent=2)
#                 with open(save_dir / "episode_values.json", "w") as f:
#                     json.dump(episode_value_means, f)

#     except KeyboardInterrupt:
#         print("\nTraining interrupted. Saving progress...")
#         if save_dir and save_models_fn:
#             save_models_fn(save_dir, actors, critic)

#     return episode_rewards, episode_task_stats, episode_value_means, critic_losses, actor_losses