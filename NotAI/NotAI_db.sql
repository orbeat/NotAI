drop table NotAI_game3 cascade constraint purge;
drop sequence NotAI_game3_seq;

drop table NotAI_Control3 cascade constraint purge;
drop sequence NotAI_Control3_seq;


create table NotAI_game3(
	ng_no number(10) primary key,
	ng_start_time date not null,			-- 게임 시작 시간
	ng_start_clock number(13,4) not null,	-- 게임 시작 클럭
	ng_end_time date not null,				-- 게임 종료 시간
	ng_end_clock number(13,4) not null		-- 게임 종료 클럭
);
create sequence NotAI_game3_seq;


create table NotAI_Control3
(
	nc_no number(13) primary key,
	nc_start_clock number(13,4) not null,		-- 누르기 시작한 시간
	
	nc_key_z number(1) not null,		-- z키를 눌렀으면 1, 아니면 0
	nc_key_x number(1) not null,		-- x키를 눌렀으면 1, 아니면 0
	nc_key_left number(1) not null,		-- left키를 눌렀으면 1, 아니면 0
	nc_key_right number(1) not null,	-- right키를 눌렀으면 1, 아니면 0
	nc_key_down number(1) not null,		-- down키를 눌렀으면 1, 아니면 0
	
	nc_score number(6) not null,				-- 누르기 시작한 시간(캡쳐한 시각)의 점수
	nc_level number(3) not null,				-- 누르기 시작한 시간(캡쳐한 시각)의 레벨
	nc_line number(4) not null,					-- 누르기 시작한 시간(캡쳐한 시각)의 부순 줄 수
	nc_next_block varchar(1 char),				-- 누르기 시작한 시간(캡쳐한 시각)의 다음 블럭
	nc_ng_no number(10) not null,				-- 해당 게임의 번호
	constraint nc_key3
		foreign key(nc_ng_no) references NotAI_game3(ng_no)
		on delete cascade
);
create sequence NotAI_Control3_seq;

select * from NotAI_game3;
select * from NotAI_control3;

select ng_no, max(nc_no)-min(nc_no)+1, max(nc_score), max(nc_level), max(nc_line) -- 게임 번호, 진행 프레임 수, 최종 점수, 최종 레벨, 최종 라인수
from NotAI_game3, NotAI_Control3
where ng_no = nc_ng_no
group by ng_no
order by ng_no;

