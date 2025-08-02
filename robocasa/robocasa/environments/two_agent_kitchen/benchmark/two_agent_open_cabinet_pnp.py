from robocasa.environments.two_agent_kitchen.two_agent_kitchen import *


class TwoAgentOpenCabinetPnP(TwoAgentKitchen):
    EXCLUDE_LAYOUTS = [8]
    def __init__(
        self, 
        obj_groups="vegetable", 
        exclude_obj_groups=None,
        *args, 
        **kwargs
    ):
        
        kwargs["layout_ids"] = [0, 10] # need singlecabinet with door handle on the rght, [0, 10] has 50% chance
        self.obj_groups = obj_groups
        self.exclude_obj_groups = exclude_obj_groups
        super().__init__(*args, **kwargs)
    
    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.cab = self.register_fixture_ref("cab", dict(id=FixtureType.DOOR_TOP_HINGE_SINGLE)) # if single, may refer to microwave
        self.microwave = self.register_fixture_ref(
            "microwave", dict(id=FixtureType.MICROWAVE),
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.microwave),
        )

        self.init_robot_base_pos = []
        self.init_robot_base_pos.append(self.sink)
        self.init_robot_base_pos.append(self.cab)
    
    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        self.cab.set_door_state(min=0.0, max=0.02, env=self, rng=self.rng) # door is closed initially

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        # obj_lang = self.get_obj_lang(obj_name="vegetable")
        ep_meta["lang"] = f"pick up vegetable from counter and place it to the cabinet"
        return ep_meta
    
    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(dict(
            name="vegetable",
            obj_groups=("eggplant"),
            exclude_obj_groups=self.exclude_obj_groups,
            graspable=True, microwavable=True,
            placement=dict(
                fixture=self.counter, # counter
                sample_region_kwargs=dict(
                    ref=self.sink,
                    # TODO: loc="right"
                ),
                size=(0.30, 0.30),
                pos=("ref", -1.0),
                try_to_place_in="container",
            ),
        ))
        cfgs.append(dict(
            name="container",
            obj_groups=("plate_color"),
            placement=dict(
                fixture=self.cab,
                size=(0.05, 0.05),
                ensure_object_boundary_in_range=False,
            ),
        ))

        return cfgs

    def _check_success(self):
        obj = self.objects["vegetable"]
        container = self.objects["container"]
        door_state = self.cab.get_door_state(env=self)
        
        task0_success = True
        for joint_p in door_state.values():
            if joint_p < 0.90:
                task0_success = False
                break

        obj_container_contact = self.check_contact(obj, container)
        container_cab_contact = self.check_contact(container, self.cab)
        gripper_obj_far = OU.gripper_obj_far(self, obj_name="vegetable")
        task1_success =  obj_container_contact and container_cab_contact and gripper_obj_far # return a list maybe work

        task_success = task0_success and task1_success
        return {'task': task_success, 'task0': task0_success, 'task1': task1_success}