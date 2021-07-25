create table NotAI_game2(
	ng_no number(10) primary key,
	ng_start_time date not null,			-- 게임 시작 시간
	ng_start_clock number(13,4) not null,	-- 게임 시작 클럭
	ng_end_time date not null,				-- 게임 종료 시간
	ng_end_clock number(13,4) not null		-- 게임 종료 클럭
);
create sequence NotAI_game2_seq;

drop table NotAI_game2 cascade constraint purge;
drop sequence NotAI_game2_seq;

create table NotAI_Control2
(
	nc_no number(13) primary key,
	--nc_push_key number(2) not null,			-- 누른 키
	--nc_start_clock number(13,4) not null,		-- 누르기 시작한 시간
	--nc_control_time number(6,4) not null,		-- 누른 시간(값이 1.5면 1.5초동안 누름)
	
	-- 값이 0이면 눌리지 않은 키
	nc_start_clock_z number(13,4) not null,		-- z키를 누르기 시작한 시간(clock())
	nc_start_clock_x number(13,4) not null,		-- x키를 누르기 시작한 시간(clock())
	nc_start_clock_left number(13,4) not null,	-- left키를 누르기 시작한 시간(clock())
	nc_start_clock_right number(13,4) not null,	-- right키를 누르기 시작한 시간(clock())
	nc_start_clock_down number(13,4) not null,	-- down키를 누르기 시작한 시간(clock())
	
	-- 값이 0이면 눌리지 않은 키
	nc_control_time_z number(6,4) not null,		-- z키를 누른 시간(값이 1.5면 1.5초동안 누름)
	nc_control_time_x number(6,4) not null,		-- x키를 누른 시간(값이 1.5면 1.5초동안 누름)
	nc_control_time_left number(6,4) not null,	-- left키를 누른 시간(값이 1.5면 1.5초동안 누름)
	nc_control_time_right number(6,4) not null,	-- right키를 누른 시간(값이 1.5면 1.5초동안 누름)
	nc_control_time_down number(6,4) not null,	-- down키를 누른 시간(값이 1.5면 1.5초동안 누름)
	
	nc_score number(6) not null,				-- 누르기 시작한 시간(캡쳐한 시각)의 점수
	nc_level number(3) not null,				-- 누르기 시작한 시간(캡쳐한 시각)의 레벨
	nc_line number(4) not null,					-- 누르기 시작한 시간(캡쳐한 시각)의 부순 줄 수
	nc_next_block varchar(1 char),				-- 누르기 시작한 시간(캡쳐한 시각)의 다음 블럭
	nc_ng_no number(10) not null,				-- 해당 게임의 번호
	constraint nc_key2
		foreign key(nc_ng_no) references NotAI_game2(ng_no)
		on delete cascade
);
create sequence NotAI_Control2_seq;

drop table NotAI_Control2 cascade constraint purge;
drop sequence NotAI_Control2_seq;

insert into NotAI_game2
values (NotAI_game_seq.nextval, to_date('20210713-165718', 'YYYYMMDD-HH24MISS'), 2.8610575, to_date('20210713-165720', 'YYYYMMDD-HH24MISS'), 4.8599817);

insert into NotAI_Control2
values (NotAI_Control_seq.nextval, 'x', 3.497611, 1.396458, 0, 0, 0, 'L', 13);

insert into NotAI_Control2
values (NotAI_Control_seq.nextval, 'right', 2.514763, 0.928671, 0, 0, 0, 'L', 13);

select * from NotAI_game2;
select * from NotAI_control2;