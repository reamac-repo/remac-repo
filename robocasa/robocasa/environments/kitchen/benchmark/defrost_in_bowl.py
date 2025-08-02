from robocasa.environments.kitchen.kitchen import *


class DefrostInBowl(Kitchen):
    def __init__(self, *args, **kwargs):

        self.water_time = 0 # used in record time for water on
        kwargs["style_ids"] = [1, 5, 6, 9, 10] # for small sink
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        
        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.sink, size=(0.5, 0.5))
        )
        self.init_robot_base_pos = self.sink

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "defrost fish in the sink"
        return ep_meta
    
    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(dict(
            name="fish",
            obj_groups=("fish"),
            exclude_obj_groups=None,
            graspable=True, microwavable=True,
            placement=dict(
                fixture=self.counter, # counter
                sample_region_kwargs=dict(
                    ref=self.counter,
                    # TODO: loc="right"
                ),
                size=(0.30, 0.30),
                pos=("ref", -1.0),
                try_to_place_in="container",
            ),
        ))

        # bowl to place the vegetable in
        cfgs.append(dict(
            name="bowl",
            obj_groups="bowl",
            placement=dict(
                fixture=self.counter,
                sample_region_kwargs=dict(
                    ref=self.sink,
                    loc="left_right",
                    top_size=(0.5, 0.5)
                ),
                size=(0.3, 0.4),
                pos=("ref", -1)
            )
        ))
        return cfgs

    def _check_success(self):

        # object relation checks
        obj = "fish"
        obj_in_sink = OU.obj_inside_of(self, obj, self.sink)
        obj_in_bowl = OU.check_obj_in_receptacle(self, obj, "bowl")
        gripper_obj_far = OU.gripper_obj_far(self, obj_name=obj)

        # need turn on and off faucet to defrost
        handle_state = self.sink.get_handle_state(env=self)
        water_on = handle_state["water_on"]
        if water_on:
            self.water_time += 1
        has_watered = water_on

        task_success = obj_in_sink and obj_in_bowl and gripper_obj_far and has_watered

        return task_success