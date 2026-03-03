from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """
    Advanced Launch File for Drowsiness Detection System
    
    Features:
    - Launch arguments for driver_id
    - Network setup (IP configuration)
    - CARLA simulator support (commented)
    - Delayed startup for all nodes
    - LLM-based decision making with multi-sensor fusion
    
    Usage:
        ros2 launch drowsiness_detection_pkg drowsiness_detection_advanced.py
        ros2 launch drowsiness_detection_pkg drowsiness_detection_advanced.py driver_id:=john_doe
    """
    
    # ========================================================================
    # LAUNCH ARGUMENTS
    # ========================================================================
    
    # Launch argument: driver ID for consistent session naming
    driver_id_arg = DeclareLaunchArgument(
        "driver_id",
        default_value="maria",
        description="Driver identifier for saving session data."
    )
    
    driver_id = LaunchConfiguration("driver_id")
    
    
    # ========================================================================
    # SYSTEM SETUP
    # ========================================================================
    
    # Setup network interface BEFORE anything else
    setup_network = ExecuteProcess(
        cmd=[
            "bash", "-c",
            "sudo ip addr add 192.168.0.10/24 dev enp130s0 || true; "
            "sudo ip link set enp130s0 up"
        ],
        output="screen",
        shell=True,
    )
    
    
    # === OPTIONAL: Launch CARLA simulator ===
    # Uncomment if using CARLA data source
    # carla_process = ExecuteProcess(
    #     cmd=[
    #         "/home/user/Downloads/CARLA_0.9.16/CarlaUE4.sh",
    #         "-RenderOffScreen",
    #         "--quality-level=LOW",
    #         "--ros2",
    #     ],
    #     output="screen",
    #     shell=True,
    # )
    
    
    # ========================================================================
    # FEATURE EXTRACTION NODES
    # ========================================================================
    
    # Camera mediapipe node (extracts PERCLOS, blink metrics)
    mediapipe_node = Node(
        package="drowsiness_detection_pkg",
        executable="camera_mediapipe_node",
        name="camera_mediapipe_node",
        output="screen",
        parameters=[{"driver_id": driver_id}],
    )
    
    
    # CARLA steering node (extracts steering entropy, lane deviation)
    carla_node = Node(
        package="drowsiness_detection_pkg",
        executable="carla_manual_control",
        name="carla_manual_control",
        output="screen",
        parameters=[{"driver_id": driver_id}],
    )
    
    
    # VLM hybrid node (detects occlusion events)
    hybrid_vlm_node = Node(
        package="drowsiness_detection_pkg",
        executable="hybrid_vlm_node",
        name="hybrid_vlm_node",
        output="screen",
    )
    
    
    # ========================================================================
    # ML CLASSIFICATION NODES
    # ========================================================================
    
    # Camera ML classifier (processes camera features)
    camera_ml_node = Node(
        package="drowsiness_detection_pkg",
        executable="camera_ml_node",
        name="camera_ml_node",
        output="screen",
    )
    
    
    # CARLA ML classifier (processes steering features)
    carla_ml_node = Node(
        package="drowsiness_detection_pkg",
        executable="carla_ml_node",
        name="carla_ml_node",
        output="screen",
    )
    
    
    # ========================================================================
    # CORE DECISION & ROUTING NODES (CRITICAL!)
    # ========================================================================
    
    # [138] LLM Reasoning Engine - Makes drowsiness decisions
    integrated_llm_node = Node(
        package="drowsiness_detection_pkg",
        executable="integrated_llm_node",
        name="integrated_llm_node",
        output="screen",
        parameters=[
            {"driver_id": driver_id},
            {"ollama_endpoint": "http://localhost:11434"},
            {"ollama_model": "llama3.1:8b"},
            {"temperature": 0.3},
            {"timeout_seconds": 90},
            {"decision_interval_seconds": 60},
        ],
    )
    
    
    # [139] Alert Dispatcher - Routes decisions to actuators
    drowsiness_alert_dispatcher = Node(
        package="drowsiness_detection_pkg",
        executable="drowsiness_alert_dispatcher",
        name="drowsiness_alert_dispatcher",
        output="screen",
        parameters=[
            {"driver_id": driver_id},
            {"audio_dir": "/home/user/DROWSINESS_DETECTION/audio_alerts"},
        ],
    )
    
    
    # ========================================================================
    # ACTUATOR CONTROL NODES
    # ========================================================================
    
    # Speaker node (plays audio alerts)
    speaker_node = Node(
        package="drowsiness_detection_pkg",
        executable="speaker_node",
        name="speaker_node",
        output="screen",
    )
    
    
    # Fan controller node (controls fan speed 0-3)
    fan_node = Node(
        package="drowsiness_detection_pkg",
        executable="fan_node",
        name="fan_node",
        output="screen",
    )
    
    
    # Steering wheel vibration node (Logitech G29)
    steering_node = Node(
        package="drowsiness_detection_pkg",
        executable="steering_node",
        name="steering_node",
        output="screen",
    )
    
    
    # ========================================================================
    # MONITORING & VISUALIZATION (OPTIONAL)
    # ========================================================================
    
    # Optional: GUI for drowsiness visualization
    # Uncomment if you have drowsiness_gui node
    # drowsiness_gui = Node(
    #     package="drowsiness_detection_pkg",
    #     executable="drowsiness_gui",
    #     name="drowsiness_gui",
    #     output="screen",
    #     parameters=[{"driver_id": driver_id}],
    # )
    
    
    # ========================================================================
    # DELAYED STARTUP
    # ========================================================================
    
    # Delay ROS2 nodes until system is fully ready
    # 5 seconds delay allows network setup to complete and simulator to initialize
    delayed_nodes = TimerAction(
        period=5.0,  # seconds
        actions=[
            # Feature extraction
            mediapipe_node,
            carla_node,
            hybrid_vlm_node,
            
            # ML Classification
            camera_ml_node,
            carla_ml_node,
            
            # Decision & Routing (CRITICAL)
            integrated_llm_node,
            drowsiness_alert_dispatcher,
            
            # Actuators
            speaker_node,
            fan_node,
            steering_node,
            
            # Optional: Visualization
            # drowsiness_gui,
        ],
    )
    
    
    # ========================================================================
    # LAUNCH DESCRIPTION
    # ========================================================================
    
    return LaunchDescription([
        # 1. Declare launch arguments
        driver_id_arg,
        
        # 2. Run system setup (network configuration)
        setup_network,
        
        # 3. Start CARLA (if needed - currently commented out)
        # carla_process,
        
        # 4. Delayed startup of all ROS2 nodes
        delayed_nodes,
    ])