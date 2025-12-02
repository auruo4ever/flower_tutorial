import io
import time
from logging import INFO, DEBUG, WARNING
from pathlib import Path
from typing import Callable, Iterable, Optional
from typing import cast
import numpy as np
import random

import torch
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord
from flwr.common import log, logger
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg, Result, Strategy
from flwr.serverapp.strategy.strategy_utils import log_strategy_start_info

from flower_tutorial.rl_controller import RLController

from flwr.common import (
    ArrayRecord,
    ConfigRecord,
    Message,
    MessageType,
    MetricRecord,
    RecordDict,
    log,
)

from flwr.serverapp.strategy.strategy_utils import (
    sample_nodes
)

class CustomFedAvg(FedAvg):

    def get_qualities(
        self, server_round: int, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        # Sample nodes
        num_nodes = int(len(list(grid.get_node_ids())))
        node_ids, _ = sample_nodes(grid, self.min_available_nodes, num_nodes)
        log(
            INFO, "GET QUALITIES"
        )
        # Always inject current server round
        config["server-round"] = server_round

        # Construct messages
        record = RecordDict(
            {self.configrecord_key: config}
        )
        return self._construct_messages(record, node_ids, "train.quality_measurement")


    def get_quality_matrix(
        self,
        replies: Iterable[Message],
    ) -> np.ndarray:
        """Extract quality scores from the received Messages."""
        quality_rows: list[list[float]] = []
        reply_contents = [msg.content for msg in replies] 
        
        if replies:
            for record in reply_contents:
                client_row: list[float] = []
                for record_item in record.metric_records.values():
                    for _, value in record_item.items():
                        client_row.append(cast(float, value))
                if client_row:
                    quality_rows.append(client_row)

        return np.array(quality_rows, dtype=np.float32)
    
    
    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid, quality_matrix: np.ndarray, b: np.ndarray,
    ) -> Iterable[Message]:
        """Configure the next round of federated training."""
        
        # Do not configure federated train if fraction_train is 0.
        if self.fraction_train == 0.0:
            return []
        
        # Sample nodes
        num_nodes = int(len(list(grid.get_node_ids())) * self.fraction_train)
        sample_size = max(num_nodes, self.min_train_nodes)
                          
        #Filter clients by threshold
        eligible_indices = np.where(np.all(quality_matrix >= b, axis=1))[0]
        
        if len(eligible_indices) == 0 or len(eligible_indices) < sample_size:
            self.last_selected_clients = []
            log(WARNING, "No clients passed the threshold.")
            return []
        
        node_ids = random.sample([list(grid.get_node_ids())[i] for i in eligible_indices], sample_size)

        #node_ids, num_total = sample_nodes(grid, self.min_available_nodes, sample_size)
        log(INFO,
            "configure_train: Sampled %s nodes (IDs: %s) out of %s eligible (%s total connected).",
            len(node_ids),
            node_ids,
            len(eligible_indices),
            len(list(grid.get_node_ids())),
        )
        # Always inject current server round
        config["server-round"] = server_round
        
        # Keep track of last selected clients for RL reward computation
        self.last_selected_clients = node_ids

        # Construct messages
        record = RecordDict(
            {self.arrayrecord_key: arrays, self.configrecord_key: config}
        )
        return self._construct_messages(record, node_ids, MessageType.TRAIN)


    
    def start(
        self,
        grid: Grid,
        initial_arrays: ArrayRecord,
        num_rounds: int = 3,
        timeout: float = 3600,
        train_config: Optional[ConfigRecord] = None,
        evaluate_config: Optional[ConfigRecord] = None,
        evaluate_fn: Optional[
            Callable[[int, ArrayRecord], Optional[MetricRecord]]
        ] = None,
    ) -> Result:
        """Execute the federated learning strategy.

        Runs the complete federated learning workflow for the specified number of
        rounds, including training, evaluation, and optional centralized evaluation.

        Parameters
        ----------
        grid : Grid
            The Grid instance used to send/receive Messages from nodes executing a
            ClientApp.
        initial_arrays : ArrayRecord
            Initial model parameters (arrays) to be used for federated learning.
        num_rounds : int (default: 3)
            Number of federated learning rounds to execute.
        timeout : float (default: 3600)
            Timeout in seconds for waiting for node responses.
        train_config : ConfigRecord, optional
            Configuration to be sent to nodes during training rounds.
            If unset, an empty ConfigRecord will be used.
        evaluate_config : ConfigRecord, optional
            Configuration to be sent to nodes during evaluation rounds.
            If unset, an empty ConfigRecord will be used.
        evaluate_fn : Callable[[int, ArrayRecord], Optional[MetricRecord]], optional
            Optional function for centralized evaluation of the global model. Takes
            server round number and array record, returns a MetricRecord or None. If
            provided, will be called before the first round and after each round.
            Defaults to None.

        Returns
        -------
        Results
            Results containing final model arrays and also training metrics, evaluation
            metrics and global evaluation metrics (if provided) from all rounds.
        """
        log(INFO, "Starting %s strategy:", self.__class__.__name__)
        log_strategy_start_info(
            num_rounds, initial_arrays, train_config, evaluate_config
        )
        self.summary()
        log(INFO, "")

        # Initialize if None
        train_config = ConfigRecord() if train_config is None else train_config
        evaluate_config = ConfigRecord() if evaluate_config is None else evaluate_config
        result = Result()

        t_start = time.time()
        # Evaluate starting global parameters
        if evaluate_fn:
            res = evaluate_fn(0, initial_arrays)
            log(INFO, "Initial global evaluation results: %s", res)
            if res is not None:
                result.evaluate_metrics_serverapp[0] = res

        arrays = initial_arrays

        for current_round in range(0, num_rounds + 1):
            
            if current_round == 0:
                #ask for quality scores once before training rounds
                quality_replies = grid.send_and_receive(
                    messages=self.get_qualities(
                        current_round,
                        train_config,
                        grid,
                    ),
                    timeout=timeout,
                )
                   
                # Aggregate quality metrics
                quality_matrix = self.get_quality_matrix(quality_replies,)
                log(INFO, "\t!!Clients Quality Metric: %s", quality_matrix)   
                
                #initialize RL controller
                
                max_clients = len(grid.get_node_ids())      # number of clients in system
                num_metrics = quality_matrix.shape[1]      # columns in quality_matrix

                rl_agent = RLController(
                    num_metrics=num_metrics, max_clients=max_clients, beta1=1.0, beta2=0.05, beta3=0.05,
                    lr = train_config["lr"], 
                    )
                      
                continue
            
            
                        
            log(INFO, "")
            log(INFO, "[ROUND %s/%s]", current_round, num_rounds)
            
            #C, b = rl_agent.decide(quality_matrix)
            
            C, b = rl_agent.decide(
                quality_matrix,
                min_clients=2,
                max_clients=max_clients
            )
            self.fraction_train = C / quality_matrix.shape[0]
            
            log(INFO, f"RL selected {self.fraction_train} clients")
            log(INFO, f"RL threshold vector: {b}")


            # -----------------------------------------------------------------
            # --- TRAINING (CLIENTAPP-SIDE) -----------------------------------
            # -----------------------------------------------------------------

            # Call strategy to configure training round
            # Send messages and wait for replies
            train_replies = grid.send_and_receive(
                messages=self.configure_train(
                    current_round,
                    arrays,
                    train_config,
                    grid,
                    quality_matrix,
                    b,
                ),
                timeout=timeout,
            )

            # Aggregate train
            agg_arrays, agg_train_metrics = self.aggregate_train(
                current_round,
                train_replies,
            )

            # Log training metrics and append to history
            if agg_arrays is not None:
                result.arrays = agg_arrays
                arrays = agg_arrays
            if agg_train_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_train_metrics)
                result.train_metrics_clientapp[current_round] = agg_train_metrics

            # -----------------------------------------------------------------
            # --- EVALUATION (CLIENTAPP-SIDE) ---------------------------------
            # -----------------------------------------------------------------

            # Call strategy to configure evaluation round
            # Send messages and wait for replies
            evaluate_replies = grid.send_and_receive(
                messages=self.configure_evaluate(
                    current_round,
                    arrays,
                    evaluate_config,
                    grid,
                ),
                timeout=timeout,
            )

            # Aggregate evaluate
            agg_evaluate_metrics = self.aggregate_evaluate(
                current_round,
                evaluate_replies,
            )

            # Log training metrics and append to history
            if agg_evaluate_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_evaluate_metrics)
                result.evaluate_metrics_clientapp[current_round] = agg_evaluate_metrics
                
                
            acc_t = agg_evaluate_metrics["eval_acc"]
            reward = rl_agent.compute_reward(acc_t, self.last_selected_clients)
            stats = rl_agent.update()
            log(INFO, f"RL Reward = {reward:.4f}, Stats: {stats}")

            # -----------------------------------------------------------------
            # --- EVALUATION (SERVERAPP-SIDE) ---------------------------------
            # -----------------------------------------------------------------

            # Centralized evaluation
            if evaluate_fn:
                log(INFO, "Global evaluation")
                res = evaluate_fn(current_round, arrays)
                log(INFO, "\t└──> MetricRecord: %s", res)
                if res is not None:
                    result.evaluate_metrics_serverapp[current_round] = res

        log(INFO, "")
        log(INFO, "Strategy execution finished in %.2fs", time.time() - t_start)
        log(INFO, "")
        log(INFO, "Final results:")
        log(INFO, "")
        for line in io.StringIO(str(result)):
            log(INFO, "\t%s", line.strip("\n"))
        log(INFO, "")

        return result
    
    
    
    