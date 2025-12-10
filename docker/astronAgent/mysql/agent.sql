select 'agent DATABASE initialization started' as '';
CREATE DATABASE IF NOT EXISTS agent;

USE agent;

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for bot_config
-- ----------------------------
DROP TABLE IF EXISTS `bot_tenant`;
CREATE TABLE `bot_tenant` (
  `id` bigint(19) NOT NULL COMMENT '主键id、雪花id',
  `name` varchar(64) NOT NULL COMMENT '应用名',
  `alias_id` varchar(32) NOT NULL COMMENT '应用标识id',
  `description` varchar(255) DEFAULT NULL COMMENT '租户描述',
  `api_key` varchar(128) DEFAULT NULL COMMENT '租户api key',
  `api_secret` varchar(128) DEFAULT NULL COMMENT '租户api秘钥',
  `create_at` datetime NOT NULL COMMENT '创建时间',
  `update_at` datetime NOT NULL COMMENT '更新时间',
  PRIMARY KEY (`id`),
  UNIQUE KEY `union_alias_id` (`alias_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


DROP TABLE IF EXISTS `bot`;
CREATE TABLE `bot` (
  `id` bigint(19) NOT NULL COMMENT '主键id、雪花id',
  `app_id` varchar(32) NOT NULL COMMENT '租户应用标识',
  `dsl` TEXT NOT NULL COMMENT '助手编排协议',
  `pub_status` bigint(8) NOT NULL COMMENT '助手当前状态：0-草稿、1-已发布',
  `create_at` datetime NOT NULL COMMENT '创建时间',
  `update_at` datetime NOT NULL COMMENT '更新时间',
  PRIMARY KEY (`id`),
  KEY `union_app_id` (`app_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


DROP TABLE IF EXISTS `bot_release`;
CREATE TABLE `bot_release` (
  `id` bigint(19) NOT NULL COMMENT '主键id、雪花id',
  `version` varchar(64) NOT NULL COMMENT '版本',
  `description` varchar(255) DEFAULT NULL COMMENT '版本描述',
  `bot_id` bigint(19) NOT NULL COMMENT '业务外键、助手表主键',
  `dsl` text NOT NULL COMMENT '当前版本助手编排协议',
  `create_at` datetime NOT NULL COMMENT '创建时间',
  `update_at` datetime NOT NULL COMMENT '更新时间',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_version_bot_id` (`version`,`bot_id`),
  KEY `union_bot_id` (`bot_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

select 'agent DATABASE initialization completed' as '';