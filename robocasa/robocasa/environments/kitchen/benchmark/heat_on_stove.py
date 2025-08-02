from robocasa.environments.kitchen.kitchen import *


class HeatOnStove(Kitchen):
    def __init__(self, *args, **kwargs):

        kwargs["style_ids"] = [1, 2, 5, 6, 7] # [0, 4, 10], [3, 8, 9, 11]
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.counter = self.register_fixture_ref("counter", dict(id=FixtureType.COUNTER))
        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        
        self.init_robot_base_pos = self.sink

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = f"pick up vegetable and heat on the stove"
        return ep_meta
    
    def _reset_internal(self):
        super()._reset_internal()

        # set initial stove knobs state
        knob_dict = self.stove.knob_joints
        valid_knobs = [k for (k, v) in knob_dict.items() if v is not None]
        # valid key: ['rear_left', 'rear_right', 'front_left', 'front_right', 'center']
        # invalid key: ['rear_center', 'front_center'] for style_ids = [1, 2, 5, 6, 7]
        for knob in valid_knobs:
            self.stove.set_knob_state(mode="off", knob=knob, env=self, rng=self.rng)
    
    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(dict(
            name="pan",
            obj_groups="pan",
            graspable=True,
            placement=dict(
                fixture=self.counter,
                ensure_object_boundary_in_range=False,
                size=(0.05, 0.02),
                pos=(0, 0),
                # rotation=(2*np.pi/8, 3*np.pi/8),
                rotation=(-np.pi/4, np.pi/4),
            ),
        ))

        cfgs.append(dict(
            name="vegetable",
            obj_groups=("corn"),
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

        return cfgs
    

    def _check_obj_location_on_stove(self, obj_name, threshold=0.08):
        knobs_state = self.stove.get_knobs_state(env=self)
        obj = self.objects[obj_name]
        obj_pos = np.array(self.sim.data.body_xpos[self.obj_body_id[obj.name]])[0:2]
        obj_on_stove = OU.check_obj_fixture_contact(self, obj_name, self.stove)
        if obj_on_stove:
            for location, site in self.stove.burner_sites.items():
                if site is not None:
                    burner_pos = np.array(self.sim.data.get_site_xpos(site.get("name")))[0:2]
                    dist = np.linalg.norm(burner_pos - obj_pos)

                    obj_on_site = (dist < threshold)
                    knob_on = (0.35 <= np.abs(knobs_state[location]) <= 2 * np.pi - 0.35) if location in knobs_state else False

                    if obj_on_site and knob_on:
                        return location
                    
        return None

    def _check_success(self):
        
        obj = self.objects["vegetable"]
        pan = self.objects["pan"]
        obj_pan_contact = self.check_contact(obj, pan)
        pan_loc = self._check_obj_location_on_stove("pan", threshold=0.15)

        task_success = (pan_loc is not None) and OU.gripper_obj_far(self, "pan") and obj_pan_contact

        return task_success