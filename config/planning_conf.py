from spock import spock

@spock
class PlanningConf:
    percentile_exploitation: float
    frontier_depth: float
    no_nav_radius: float
    yolo_confidence: float
    yolov7_confidence: float  # Legacy YAML compatibility; no evaluation actor uses YOLOv7.
    obj_confidence: float
    filter_detections_depth: bool
    consensus_filtering: bool
    allow_replan: bool
    use_frontiers: bool
    allow_far_plan: bool
    using_ov: bool
    max_detect_distance: float
    objectnav_found_distance: float
    obstcl_kernel_size: float
    min_goal_dist: float

    # 3D frontier / semantic-memory configuration. Defaults preserve the
    # established research settings when older YAML files are loaded.
    dfine_confidence: float = 0.8
    dfine_config: str = "vision_models/dfine/configs/dfine/objects365/dfine_hgnetv2_x_obj2coco.yml"
    dfine_checkpoint: str = "weights/dfine_x_obj2coco.pth"
    single_objectnav_found_distance: float = 0.8
    enable_history_poi: bool = True
    history_poi_objectnav_match_xy_m: float = 1.0
    history_poi_objectnav_match_z_m: float = 1.0
    frontier_dirty_margin: int = 5
    frontier_missing_grace: int = 3
    frontier_normal_min_cluster_size: int = 5
    frontier_init_min_cluster_size: int = 2

    stair_sam_threshold: float = 0.689
    segformer_stair_model: str = "weights/segformer_b2_ade20k"
    segformer_stair_pixel_threshold: float = 0.25
    segformer_stair_min_area: int = 80
    segformer_stair_max_components: int = 8
    segformer_stair_fp16: bool = True
    descend_pitched_conf_threshold: float = 0.48
    descend_pitched_sam_threshold: float = 0.48
    descend_geometry_threshold: float = 0.65
    descend_wait_frames: int = 5
    ascent_proposal_threshold: float = 0.55
    ascent_strong_threshold: float = 0.80
    ascent_required_seen: int = 2
    ascent_candidate_ttl: int = 10
    ascent_approach_distance: float = 0.60
    stair_force_forward_steps: int = 3
    stair_min_descent_layers: int = 3
    stair_descent_floor_confirm_frames: int = 3
    stair_descent_frontier_guard_distance: float = 0.30
    stair_min_ascent_layers: int = 6
    stair_ascent_floor_confirm_frames: int = 3
    stair_sem_session_radius_m: float = 2.5
    stair_sem_xy_refine_delay_frames: int = 3
    stair_replan_period: int = 3

    objectnav_miss_limit: int = 5
    objectnav_edge_oscillation_found_cycles: int = 5
    object_found_min_confidence: float = 0.7
    object_found_min_box_area_ratio: float = 0.015625
    object_found_min_mask_voxels: int = 10
    object_found_edge_margin_px: float = 8.0
    object_found_max_center_offset: float = 0.75
    # Absolute frame-local SED check only at the final called_found output.
    enable_sed_found_confirmation: bool = True
    sed_found_threshold: float = 0.15
    sed_reject_cooldown_frames: int = 50
    sed_reject_radius_xy_m: float = 0.5
    sed_reject_radius_z_m: float = 0.6
