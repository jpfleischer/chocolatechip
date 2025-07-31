-- MySQL dump 10.13  Distrib 8.0.42, for Linux (x86_64)
--
-- Host: localhost    Database: testdb
-- ------------------------------------------------------
-- Server version	8.0.42-0ubuntu0.20.04.1

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Current Database: `testdb`
--

CREATE DATABASE /*!32312 IF NOT EXISTS*/ `testdb` /*!40100 DEFAULT CHARACTER SET latin1 */ /*!80016 DEFAULT ENCRYPTION='N' */;

USE `testdb`;

--
-- Table structure for table `CentroidsTable`
--

DROP TABLE IF EXISTS `CentroidsTable`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `CentroidsTable` (
  `track_id` bigint DEFAULT NULL,
  `start_timestamp` datetime DEFAULT NULL,
  `end_timestamp` datetime(6) DEFAULT NULL,
  `x` float DEFAULT NULL,
  `y` float DEFAULT NULL,
  `class` varchar(20) DEFAULT NULL,
  `camera_id` int DEFAULT NULL,
  `intersection_id` int DEFAULT NULL,
  `path` varchar(20) DEFAULT NULL,
  `movement` varchar(20) DEFAULT NULL,
  `phase` int DEFAULT NULL,
  `name` varchar(50) DEFAULT NULL,
  `ordering` int DEFAULT NULL,
  `version` int NOT NULL,
  KEY `intersection_id` (`intersection_id`) USING BTREE,
  KEY `camera_id` (`camera_id`) USING BTREE,
  KEY `phase` (`phase`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `CentroidsTablePrev`
--

DROP TABLE IF EXISTS `CentroidsTablePrev`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `CentroidsTablePrev` (
  `track_id` bigint DEFAULT NULL,
  `start_timestamp` datetime DEFAULT NULL,
  `end_timestamp` datetime(6) DEFAULT NULL,
  `x` float DEFAULT NULL,
  `y` float DEFAULT NULL,
  `class` varchar(20) DEFAULT NULL,
  `camera_id` int DEFAULT NULL,
  `intersection_id` int DEFAULT NULL,
  `path` varchar(20) DEFAULT NULL,
  `movement` varchar(20) DEFAULT NULL,
  `phase` int DEFAULT NULL,
  `name` varchar(50) DEFAULT NULL,
  `ordering` int DEFAULT NULL,
  `version` int NOT NULL,
  KEY `intersection_id` (`intersection_id`) USING BTREE,
  KEY `camera_id` (`camera_id`) USING BTREE,
  KEY `phase` (`phase`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `Events`
--

DROP TABLE IF EXISTS `Events`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `Events` (
  `intersection_id` int DEFAULT NULL,
  `camera_id` int DEFAULT NULL,
  `timestamp` datetime(6) DEFAULT NULL,
  `dow` int DEFAULT NULL,
  `hod` int DEFAULT NULL,
  `frame_id` int DEFAULT NULL,
  `conflict_x` float DEFAULT NULL,
  `conflict_y` float DEFAULT NULL,
  `unique_ID1` bigint DEFAULT NULL,
  `unique_ID2` bigint DEFAULT NULL,
  `class1` varchar(20) DEFAULT NULL,
  `class2` varchar(20) DEFAULT NULL,
  `phase1` int DEFAULT NULL,
  `phase2` int DEFAULT NULL,
  `time` float DEFAULT NULL,
  `bb_time` float DEFAULT NULL,
  `ttc_rank` float DEFAULT NULL,
  `p2v` tinyint(1) DEFAULT NULL,
  `city` varchar(20) DEFAULT NULL,
  `state` varchar(10) DEFAULT NULL,
  `cluster1` varchar(30) DEFAULT NULL,
  `cluster2` varchar(30) DEFAULT NULL,
  `is_conflicting` float DEFAULT NULL,
  `speed1` float DEFAULT NULL,
  `speed2` float DEFAULT NULL,
  `distance` float DEFAULT NULL,
  `bb_distance` float DEFAULT NULL,
  `deceleration1` float DEFAULT NULL,
  `deceleration2` float DEFAULT NULL,
  `decel1_ts` datetime(6) DEFAULT NULL,
  `decel2_ts` datetime(6) DEFAULT NULL,
  `type` int DEFAULT NULL,
  `signal_state` varchar(20) DEFAULT NULL,
  `phase_duration` float DEFAULT NULL,
  `percent_in_phase` float DEFAULT NULL,
  `ped_signal_state` varchar(20) DEFAULT NULL,
  `ped_phase_duration` float DEFAULT NULL,
  `ped_percent_in_phase` float DEFAULT NULL,
  `cpi1` float DEFAULT NULL,
  `cpi2` float DEFAULT NULL,
  `angle` int DEFAULT NULL,
  `cycle` int DEFAULT NULL,
  `num_involved` int DEFAULT NULL,
  `intersection_diagonal` float DEFAULT NULL,
  `median_width` float DEFAULT NULL,
  `total_vehicles` int DEFAULT NULL,
  `conflict_type` int DEFAULT NULL,
  `include_flag` int DEFAULT NULL,
  `UID_Incompat` tinyint(1) DEFAULT NULL,
  `event_ID` int DEFAULT NULL,
  `annotation` int DEFAULT NULL,
  `clip_command` text,
  `date_processed` datetime(6) DEFAULT NULL,
  `notes` text,
  `human_annotation` varchar(20) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `ExtendedPedLanes`
--

DROP TABLE IF EXISTS `ExtendedPedLanes`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `ExtendedPedLanes` (
  `timestamp` datetime DEFAULT NULL,
  `intersection_id` int DEFAULT NULL,
  `camera_id` int DEFAULT NULL,
  `name` varchar(20) DEFAULT NULL,
  `p1_x` int DEFAULT NULL,
  `p1_y` int DEFAULT NULL,
  `p2_x` int DEFAULT NULL,
  `p2_y` int DEFAULT NULL,
  `mapdirection` varchar(20) DEFAULT NULL,
  KEY `intersection_id` (`intersection_id`) USING BTREE,
  KEY `camera_id` (`camera_id`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `HCSRatio`
--

DROP TABLE IF EXISTS `HCSRatio`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `HCSRatio` (
  `intersection_id` int DEFAULT NULL,
  `detector` int DEFAULT NULL,
  `direction` varchar(10) DEFAULT NULL,
  `ratio` float DEFAULT NULL,
  `city` varchar(20) DEFAULT NULL,
  KEY `intersection_id` (`intersection_id`) USING BTREE,
  KEY `detector` (`detector`) USING BTREE,
  KEY `direction` (`direction`) USING BTREE,
  KEY `city` (`city`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `HeatMap`
--

DROP TABLE IF EXISTS `HeatMap`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `HeatMap` (
  `time` float DEFAULT NULL,
  `frame_id` int NOT NULL DEFAULT '0',
  `track_id` int NOT NULL DEFAULT '0',
  `center_x` float DEFAULT NULL,
  `center_y` float DEFAULT NULL,
  `class` varchar(20) DEFAULT NULL,
  `timestamp` datetime DEFAULT NULL,
  `intersection_id` int DEFAULT NULL,
  `camera_id` int DEFAULT NULL,
  `unique_ID` bigint NOT NULL DEFAULT '0',
  `small` int DEFAULT NULL,
  `medium` int DEFAULT NULL,
  `large` int DEFAULT NULL,
  PRIMARY KEY (`frame_id`,`track_id`,`unique_ID`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `IntersectionProperties`
--

DROP TABLE IF EXISTS `IntersectionProperties`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `IntersectionProperties` (
  `intersection_id` int DEFAULT NULL,
  `camera_id` int DEFAULT NULL,
  `city` varchar(30) DEFAULT NULL,
  `state` varchar(5) DEFAULT NULL,
  `country` varchar(20) DEFAULT NULL,
  `speedlimit` int DEFAULT NULL,
  `objects` varchar(100) DEFAULT NULL,
  `phase2` varchar(5) DEFAULT NULL,
  `south_north` varchar(50) DEFAULT NULL,
  `north_south` varchar(50) DEFAULT NULL,
  `east_west` varchar(50) DEFAULT NULL,
  `west_east` varchar(50) DEFAULT NULL,
  `phase2stopbar` varchar(50) DEFAULT NULL,
  `phase4stopbar` varchar(50) DEFAULT NULL,
  `phase6stopbar` varchar(50) DEFAULT NULL,
  `phase8stopbar` varchar(50) DEFAULT NULL,
  `permittedright` varchar(20) DEFAULT NULL,
  `through` float DEFAULT NULL,
  `curved` float DEFAULT NULL,
  `pathlength` float DEFAULT NULL,
  `median_2` int DEFAULT NULL,
  `median_4` int DEFAULT NULL,
  `median_6` int DEFAULT NULL,
  `median_8` int DEFAULT NULL,
  `pixels2meter` float DEFAULT NULL,
  `polygon` varchar(300) DEFAULT NULL,
  `verticalgrid` varchar(300) DEFAULT NULL,
  `horizontalgrid` varchar(300) DEFAULT NULL,
  `c1` varchar(50) DEFAULT NULL,
  `c2` varchar(50) DEFAULT NULL,
  `c3` varchar(50) DEFAULT NULL,
  `c4` varchar(100) DEFAULT NULL,
  `c5` varchar(100) DEFAULT NULL,
  `c6` varchar(100) DEFAULT NULL,
  KEY `intersection_id` (`intersection_id`) USING BTREE,
  KEY `camera_id` (`camera_id`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `Lanes`
--

DROP TABLE IF EXISTS `Lanes`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `Lanes` (
  `timestamp` datetime DEFAULT NULL,
  `intersection_id` int DEFAULT NULL,
  `camera_id` int DEFAULT NULL,
  `approach_id` int DEFAULT NULL,
  `name` varchar(20) DEFAULT NULL,
  `p1_x` int DEFAULT NULL,
  `p1_y` int DEFAULT NULL,
  `p2_x` int DEFAULT NULL,
  `p2_y` int DEFAULT NULL,
  `width` float DEFAULT NULL,
  `mapdirection` varchar(20) DEFAULT NULL,
  `trafficdirection` varchar(20) DEFAULT NULL,
  `detector` int DEFAULT NULL,
  KEY `intersection_id` (`intersection_id`) USING BTREE,
  KEY `camera_id` (`camera_id`) USING BTREE,
  KEY `approach_id` (`approach_id`) USING BTREE,
  KEY `trafficdirection` (`trafficdirection`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `LidarTracks`
--

DROP TABLE IF EXISTS `LidarTracks`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `LidarTracks` (
  `time` double DEFAULT NULL,
  `frame_id` bigint DEFAULT NULL,
  `track_id` bigint DEFAULT NULL,
  `center_x` double DEFAULT NULL,
  `center_y` double DEFAULT NULL,
  `class` text,
  `timestamp` datetime(6) DEFAULT NULL,
  `intersection_id` bigint DEFAULT NULL,
  `camera_id` bigint DEFAULT NULL,
  `nearmiss` bigint DEFAULT NULL,
  `signature` bigint DEFAULT NULL,
  `unique_ID` bigint DEFAULT NULL,
  `raw_x` double DEFAULT NULL,
  `raw_y` double DEFAULT NULL,
  `b1` bigint DEFAULT NULL,
  `b2` bigint DEFAULT NULL,
  `b3` bigint DEFAULT NULL,
  `i1` bigint DEFAULT NULL,
  `i2` bigint DEFAULT NULL,
  `i3` bigint DEFAULT NULL,
  `r1` double DEFAULT NULL,
  `r2` double DEFAULT NULL,
  `r3` double DEFAULT NULL,
  `c1` text,
  `c2` text,
  `c3` text,
  KEY `unique_ID` (`unique_ID`) USING BTREE,
  KEY `track_id` (`track_id`) USING BTREE,
  KEY `timestamp` (`timestamp`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `LoopVideoActivations`
--

DROP TABLE IF EXISTS `LoopVideoActivations`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `LoopVideoActivations` (
  `intersection_id` int DEFAULT NULL,
  `camera_id` int DEFAULT NULL,
  `city` varchar(20) DEFAULT NULL,
  `timestamp` datetime(6) DEFAULT NULL,
  `approach` varchar(10) DEFAULT NULL,
  `lane` int DEFAULT NULL,
  `video` int DEFAULT NULL,
  `atspm` int DEFAULT NULL,
  KEY `intersection_id` (`intersection_id`) USING BTREE,
  KEY `camera_id` (`camera_id`) USING BTREE,
  KEY `timestamp` (`timestamp`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `LowSpeedTable`
--

DROP TABLE IF EXISTS `LowSpeedTable`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `LowSpeedTable` (
  `intersection_id` int DEFAULT NULL,
  `camera_id` int DEFAULT NULL,
  `timestamp` datetime DEFAULT NULL,
  `center_x` float DEFAULT NULL,
  `center_y` float DEFAULT NULL,
  `speed` float DEFAULT NULL,
  `class` varchar(20) DEFAULT NULL,
  `phase` int DEFAULT NULL,
  `unique_ID` bigint DEFAULT NULL,
  `city` varchar(20) DEFAULT NULL,
  `cluster` varchar(30) DEFAULT NULL,
  KEY `intersection_id` (`intersection_id`) USING BTREE,
  KEY `timestamp` (`timestamp`) USING BTREE,
  KEY `phase` (`phase`) USING BTREE,
  KEY `class` (`class`) USING BTREE,
  KEY `city` (`city`) USING BTREE,
  KEY `cluster` (`cluster`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `OffsetTable`
--

DROP TABLE IF EXISTS `OffsetTable`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `OffsetTable` (
  `intersection_id` int DEFAULT NULL,
  `camera_id` int DEFAULT NULL,
  `year` int DEFAULT NULL,
  `month` int DEFAULT NULL,
  `day` int DEFAULT NULL,
  `hour` int DEFAULT NULL,
  `minute` int DEFAULT NULL,
  `city` varchar(20) DEFAULT NULL,
  `state` varchar(20) DEFAULT NULL,
  `offset` varchar(30) DEFAULT NULL,
  `timestamp` datetime(6) DEFAULT NULL,
  KEY `intersection_id` (`intersection_id`) USING BTREE,
  KEY `year` (`year`) USING BTREE,
  KEY `month` (`month`) USING BTREE,
  KEY `day` (`day`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `OnlineSPaT`
--

DROP TABLE IF EXISTS `OnlineSPaT`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `OnlineSPaT` (
  `intersection_id` int DEFAULT NULL,
  `camera_id` int DEFAULT NULL,
  `timestamp` datetime(3) DEFAULT NULL,
  `hexphase` varchar(6) DEFAULT NULL,
  `cycle` int DEFAULT NULL,
  `city` varchar(20) DEFAULT NULL,
  `state` varchar(20) DEFAULT NULL,
  `type` int DEFAULT NULL,
  KEY `intersection_id` (`intersection_id`) USING BTREE,
  KEY `timestamp` (`timestamp`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `PETtable`
--

DROP TABLE IF EXISTS `PETtable`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `PETtable` (
  `intersection_id` int DEFAULT NULL,
  `camera_id` int DEFAULT NULL,
  `timestamp1` datetime DEFAULT NULL,
  `timestamp2` datetime DEFAULT NULL,
  `conflict_x` float DEFAULT NULL,
  `conflict_y` float DEFAULT NULL,
  `unique_ID1` bigint DEFAULT NULL,
  `unique_ID2` bigint DEFAULT NULL,
  `class1` varchar(20) DEFAULT NULL,
  `class2` varchar(20) DEFAULT NULL,
  `phase1` int DEFAULT NULL,
  `phase2` int DEFAULT NULL,
  `pet` float DEFAULT NULL,
  `p2v` tinyint(1) DEFAULT NULL,
  `city` varchar(20) DEFAULT NULL,
  `cluster1` varchar(20) DEFAULT NULL,
  `cluster2` varchar(20) DEFAULT NULL,
  `speed1` float DEFAULT NULL,
  `speed2` float DEFAULT NULL,
  KEY `intersection_id` (`intersection_id`) USING BTREE,
  KEY `camera_id` (`camera_id`) USING BTREE,
  KEY `timestamp1` (`timestamp1`) USING BTREE,
  KEY `phase1` (`phase1`) USING BTREE,
  KEY `phase2` (`phase2`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `Paths`
--

DROP TABLE IF EXISTS `Paths`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `Paths` (
  `timestamp` datetime DEFAULT NULL,
  `intersection_id` int DEFAULT NULL,
  `camera_id` int DEFAULT NULL,
  `phase` int DEFAULT NULL,
  `name` varchar(20) DEFAULT NULL,
  `ingressLane` varchar(20) DEFAULT NULL,
  `egressLane` varchar(20) DEFAULT NULL,
  `movement` varchar(20) DEFAULT NULL,
  `source` int DEFAULT NULL,
  KEY `intersection_id` (`intersection_id`) USING BTREE,
  KEY `camera_id` (`camera_id`) USING BTREE,
  KEY `phase` (`phase`) USING BTREE,
  KEY `movement` (`movement`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `PedLanes`
--

DROP TABLE IF EXISTS `PedLanes`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `PedLanes` (
  `timestamp` datetime DEFAULT NULL,
  `intersection_id` int DEFAULT NULL,
  `camera_id` int DEFAULT NULL,
  `name` varchar(20) DEFAULT NULL,
  `p1_x` int DEFAULT NULL,
  `p1_y` int DEFAULT NULL,
  `p2_x` int DEFAULT NULL,
  `p2_y` int DEFAULT NULL,
  `mapdirection` varchar(20) DEFAULT NULL,
  KEY `intersection_id` (`intersection_id`) USING BTREE,
  KEY `camera_id` (`camera_id`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `PedPaths`
--

DROP TABLE IF EXISTS `PedPaths`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `PedPaths` (
  `timestamp` datetime DEFAULT NULL,
  `intersection_id` int DEFAULT NULL,
  `camera_id` int DEFAULT NULL,
  `phase` int DEFAULT NULL,
  `name` varchar(20) DEFAULT NULL,
  `ingressLane` varchar(20) DEFAULT NULL,
  `egressLane` varchar(20) DEFAULT NULL,
  `movement` varchar(20) DEFAULT NULL,
  `source` int DEFAULT NULL,
  KEY `intersection_id` (`intersection_id`) USING BTREE,
  KEY `camera_id` (`camera_id`) USING BTREE,
  KEY `phase` (`phase`) USING BTREE,
  KEY `movement` (`movement`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `RawLidarInfo`
--

DROP TABLE IF EXISTS `RawLidarInfo`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `RawLidarInfo` (
  `unique_ID` bigint NOT NULL,
  `track_id` int DEFAULT NULL,
  `x` float DEFAULT NULL,
  `y` float DEFAULT NULL,
  `z` float DEFAULT NULL,
  `h` float DEFAULT NULL,
  `dx` float DEFAULT NULL,
  `dy` float DEFAULT NULL,
  `dz` float DEFAULT NULL,
  `frame_name` varchar(80) DEFAULT NULL,
  `timestamp` datetime(6) NOT NULL,
  `type_name` varchar(20) DEFAULT NULL,
  `camera_id` int NOT NULL,
  KEY `unique_ID` (`unique_ID`),
  KEY `timestamp` (`timestamp`),
  KEY `camera_id` (`camera_id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `RealDisplayInfo`
--

DROP TABLE IF EXISTS `RealDisplayInfo`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `RealDisplayInfo` (
  `frame_id` int DEFAULT NULL,
  `track_id` int DEFAULT NULL,
  `center_x` float DEFAULT NULL,
  `center_y` float DEFAULT NULL,
  `class` varchar(20) DEFAULT NULL,
  `timestamp` datetime(3) DEFAULT NULL,
  `intersection_id` int DEFAULT NULL,
  `camera_id` int DEFAULT NULL,
  `SPAT` varchar(6) DEFAULT NULL,
  `Cycle` int DEFAULT NULL,
  `speed_x` float DEFAULT NULL,
  `speed_y` float DEFAULT NULL,
  `unique_ID` bigint DEFAULT NULL,
  `skip_begin` tinyint(1) DEFAULT NULL,
  `skip_end` tinyint(1) DEFAULT NULL,
  `skip_mid` tinyint(1) DEFAULT NULL,
  `skip_angle` tinyint(1) DEFAULT NULL,
  `raw_x` float DEFAULT NULL,
  `raw_y` float DEFAULT NULL,
  `b1` tinyint(1) DEFAULT NULL,
  `b2` tinyint(1) DEFAULT NULL,
  `b3` tinyint(1) DEFAULT NULL,
  `i1` int DEFAULT NULL,
  `i2` int DEFAULT NULL,
  `i3` int DEFAULT NULL,
  `r1` float DEFAULT NULL,
  `r2` float DEFAULT NULL,
  `r3` float DEFAULT NULL,
  `c1_x` float DEFAULT NULL,
  `c1_y` float DEFAULT NULL,
  `c2_x` float DEFAULT NULL,
  `c2_y` float DEFAULT NULL,
  `c3_x` float DEFAULT NULL,
  `c3_y` float DEFAULT NULL,
  `c4_x` float DEFAULT NULL,
  `c4_y` float DEFAULT NULL,
  `c1` varchar(20) DEFAULT NULL,
  `c2` varchar(20) DEFAULT NULL,
  `c3` varchar(20) DEFAULT NULL,
  KEY `track_id` (`track_id`) USING BTREE,
  KEY `frame_id` (`frame_id`) USING BTREE,
  KEY `unique_ID` (`unique_ID`),
  KEY `timestamp` (`timestamp`),
  KEY `ts_search` (`camera_id`,`timestamp`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `RealTrackProperties`
--

DROP TABLE IF EXISTS `RealTrackProperties`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `RealTrackProperties` (
  `intersection_id` int DEFAULT NULL,
  `camera_id` int NOT NULL,
  `timestamp` datetime(3) DEFAULT NULL,
  `track_id` int DEFAULT NULL,
  `class` varchar(20) DEFAULT NULL,
  `phase` int DEFAULT NULL,
  `lane` varchar(20) DEFAULT NULL,
  `movement` varchar(20) DEFAULT NULL,
  `cluster` varchar(50) DEFAULT NULL,
  `isAnomalous` tinyint(1) DEFAULT NULL,
  `redJump` tinyint(1) DEFAULT NULL,
  `nearmiss` tinyint(1) DEFAULT NULL,
  `signature` int DEFAULT NULL,
  `unique_ID` bigint NOT NULL DEFAULT '0',
  `small` tinyint(1) DEFAULT NULL,
  `outside` tinyint(1) DEFAULT NULL,
  `hispeed` tinyint(1) DEFAULT NULL,
  `oddshape` tinyint(1) DEFAULT NULL,
  `realano` tinyint(1) DEFAULT NULL,
  `lanechange` tinyint(1) DEFAULT NULL,
  `wrongway` tinyint(1) DEFAULT NULL,
  `redViolation` tinyint(1) DEFAULT NULL,
  `yellowJump` tinyint(1) DEFAULT NULL,
  `i1` int DEFAULT NULL,
  `i2` int DEFAULT NULL,
  `i3` int DEFAULT NULL,
  `r1` float DEFAULT NULL,
  `r2` float DEFAULT NULL,
  `r3` float DEFAULT NULL,
  `c1` varchar(20) DEFAULT NULL,
  `c2` varchar(20) DEFAULT NULL,
  `c3` varchar(20) DEFAULT NULL,
  `city` varchar(20) DEFAULT NULL,
  PRIMARY KEY (`camera_id`,`unique_ID`),
  KEY `timestamp` (`timestamp`) USING BTREE,
  KEY `track_id` (`track_id`) USING BTREE,
  KEY `ps` (`phase`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `TTCPET`
--

DROP TABLE IF EXISTS `TTCPET`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `TTCPET` (
  `intersection_id` int DEFAULT NULL,
  `camera_id` int DEFAULT NULL,
  `city` varchar(20) DEFAULT NULL,
  `timestamp` datetime(6) DEFAULT NULL,
  `timestamp_decettc` datetime(6) DEFAULT NULL,
  `timestamp_petobj2` datetime(6) DEFAULT NULL,
  `conflict_x` float DEFAULT NULL,
  `conflict_y` float DEFAULT NULL,
  `unique_ID1` bigint DEFAULT NULL,
  `unique_ID2` bigint DEFAULT NULL,
  `object_type1` varchar(20) DEFAULT NULL,
  `object_type2` varchar(20) DEFAULT NULL,
  `phase1` int DEFAULT NULL,
  `phase2` int DEFAULT NULL,
  `EventType` int DEFAULT NULL,
  `time` float DEFAULT NULL,
  `cluster1` varchar(20) DEFAULT NULL,
  `cluster2` varchar(2) DEFAULT NULL,
  `speed1` float DEFAULT NULL,
  `speed2` float DEFAULT NULL,
  `signal_phase` varchar(20) DEFAULT NULL,
  `percent_in_phase` float DEFAULT NULL,
  KEY `intersection_id` (`intersection_id`) USING BTREE,
  KEY `camera_id` (`camera_id`) USING BTREE,
  KEY `unique_ID1` (`unique_ID1`) USING BTREE,
  KEY `unique_ID2` (`unique_ID2`) USING BTREE,
  KEY `timestamp` (`timestamp`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `TTCTable`
--

DROP TABLE IF EXISTS `TTCTable`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `TTCTable` (
  `intersection_id` int DEFAULT NULL,
  `camera_id` int DEFAULT NULL,
  `timestamp` datetime(6) DEFAULT NULL,
  `dow` int DEFAULT NULL,
  `hod` int DEFAULT NULL,
  `frame_id` int DEFAULT NULL,
  `conflict_x` float DEFAULT NULL,
  `conflict_y` float DEFAULT NULL,
  `unique_ID1` bigint DEFAULT NULL,
  `unique_ID2` bigint DEFAULT NULL,
  `class1` varchar(20) DEFAULT NULL,
  `class2` varchar(20) DEFAULT NULL,
  `phase1` int DEFAULT NULL,
  `phase2` int DEFAULT NULL,
  `time` float DEFAULT NULL,
  `bb_time` float DEFAULT NULL,
  `ttc_rank` float DEFAULT NULL,
  `p2v` tinyint(1) DEFAULT NULL,
  `city` varchar(20) DEFAULT NULL,
  `state` varchar(10) DEFAULT NULL,
  `cluster1` varchar(30) DEFAULT NULL,
  `cluster2` varchar(30) DEFAULT NULL,
  `is_conflicting` float DEFAULT NULL,
  `speed1` float DEFAULT NULL,
  `speed2` float DEFAULT NULL,
  `distance` float DEFAULT NULL,
  `bb_distance` float DEFAULT NULL,
  `deceleration1` float DEFAULT NULL,
  `deceleration2` float DEFAULT NULL,
  `decel1_ts` datetime(6) DEFAULT NULL,
  `decel2_ts` datetime(6) DEFAULT NULL,
  `type` int DEFAULT NULL,
  `signal_state` varchar(20) DEFAULT NULL,
  `phase_duration` float DEFAULT NULL,
  `percent_in_phase` float DEFAULT NULL,
  `ped_signal_state` varchar(20) DEFAULT NULL,
  `ped_phase_duration` float DEFAULT NULL,
  `ped_percent_in_phase` float DEFAULT NULL,
  `cpi1` float DEFAULT NULL,
  `cpi2` float DEFAULT NULL,
  `angle` int DEFAULT NULL,
  `cycle` int DEFAULT NULL,
  `num_involved` int DEFAULT NULL,
  `intersection_diagonal` float DEFAULT NULL,
  `median_width` float DEFAULT NULL,
  `total_vehicles` int DEFAULT NULL,
  `conflict_type` int DEFAULT '0',
  `include_flag` int DEFAULT '0',
  KEY `intersection_id` (`intersection_id`) USING BTREE,
  KEY `timestamp` (`timestamp`) USING BTREE,
  KEY `phase1` (`phase1`) USING BTREE,
  KEY `phase2` (`phase2`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `TrackRange`
--

DROP TABLE IF EXISTS `TrackRange`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `TrackRange` (
  `timestamp` datetime DEFAULT NULL,
  `intersection_id` int DEFAULT NULL,
  `camera_id` int DEFAULT NULL,
  `combined_camera_id` int DEFAULT NULL,
  `year` int DEFAULT NULL,
  `month` int DEFAULT NULL,
  `day` int DEFAULT NULL,
  `hour` int DEFAULT NULL,
  `minute` int DEFAULT NULL,
  `second` int DEFAULT NULL,
  KEY `intersection_id` (`intersection_id`) USING BTREE,
  KEY `camera_id` (`camera_id`) USING BTREE,
  KEY `minute` (`minute`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `TracksReal`
--

DROP TABLE IF EXISTS `TracksReal`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `TracksReal` (
  `track_id` bigint DEFAULT NULL,
  `Class` varchar(20) DEFAULT NULL,
  `Approach` varchar(20) DEFAULT NULL,
  `min_gap` float DEFAULT NULL,
  `max_gap` float DEFAULT NULL,
  `avg_gap` float DEFAULT NULL,
  `Responsetime` float DEFAULT NULL,
  `average_speed` float DEFAULT NULL,
  `Max_speed` float DEFAULT NULL,
  `Min_speed` float DEFAULT NULL,
  `Cluster` varchar(20) DEFAULT NULL,
  `Cycle` int DEFAULT NULL,
  `start_phase` varchar(20) DEFAULT NULL,
  `end_phase` varchar(20) DEFAULT NULL,
  `start_timestamp` datetime(3) DEFAULT NULL,
  `end_timestamp` datetime(3) DEFAULT NULL,
  `redJump` tinyint(1) DEFAULT NULL,
  `trackTurn` tinyint(1) DEFAULT NULL,
  `cpi` float DEFAULT NULL,
  `intersection_id` int DEFAULT NULL,
  `camera_id` int DEFAULT NULL,
  `city` varchar(20) DEFAULT NULL,
  KEY `intersection_id` (`intersection_id`) USING BTREE,
  KEY `camera_id` (`camera_id`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `TracksWeek`
--

DROP TABLE IF EXISTS `TracksWeek`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `TracksWeek` (
  `track_id` bigint DEFAULT NULL,
  `Class` text,
  `Approach` text,
  `min_gap` double DEFAULT NULL,
  `max_gap` double DEFAULT NULL,
  `avg_gap` double DEFAULT NULL,
  `Responsetime` double DEFAULT NULL,
  `average_speed` double DEFAULT NULL,
  `Max_speed` double DEFAULT NULL,
  `Min_speed` double DEFAULT NULL,
  `Cluster` text,
  `Cycle` bigint DEFAULT NULL,
  `start_phase` text,
  `end_phase` text,
  `start_timestamp` datetime DEFAULT NULL,
  `end_timestamp` datetime DEFAULT NULL,
  `redJump` bigint DEFAULT NULL,
  `trackTurn` bigint DEFAULT NULL,
  `intersection_id` bigint DEFAULT NULL,
  `camera_id` bigint DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `VehicleConflictTypes`
--

DROP TABLE IF EXISTS `VehicleConflictTypes`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `VehicleConflictTypes` (
  `type` varchar(30) DEFAULT NULL,
  `code` int DEFAULT NULL,
  KEY `type` (`type`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `VideoProperties`
--

DROP TABLE IF EXISTS `VideoProperties`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `VideoProperties` (
  `intersection_id` int DEFAULT NULL,
  `camera_id` int DEFAULT NULL,
  `path` varchar(100) DEFAULT NULL,
  `start` datetime(3) DEFAULT NULL,
  `end` datetime(3) DEFAULT NULL,
  KEY `intersection_id` (`intersection_id`) USING BTREE,
  KEY `camera_id` (`camera_id`) USING BTREE,
  KEY `start` (`start`) USING BTREE,
  KEY `end` (`end`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `VideoPropertiesNew`
--

DROP TABLE IF EXISTS `VideoPropertiesNew`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `VideoPropertiesNew` (
  `intersection_id` int DEFAULT NULL,
  `camera_id` int DEFAULT NULL,
  `path` varchar(255) DEFAULT NULL,
  `start` datetime DEFAULT NULL,
  `end` datetime DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `VideoTracks`
--

DROP TABLE IF EXISTS `VideoTracks`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `VideoTracks` (
  `time` double DEFAULT NULL,
  `frame_id` bigint DEFAULT NULL,
  `track_id` bigint DEFAULT NULL,
  `center_x` double DEFAULT NULL,
  `center_y` double DEFAULT NULL,
  `class` text,
  `timestamp` datetime(6) DEFAULT NULL,
  `intersection_id` bigint DEFAULT NULL,
  `camera_id` bigint DEFAULT NULL,
  `nearmiss` bigint DEFAULT NULL,
  `signature` bigint DEFAULT NULL,
  `unique_ID` bigint DEFAULT NULL,
  `raw_x` double DEFAULT NULL,
  `raw_y` double DEFAULT NULL,
  `b1` bigint DEFAULT NULL,
  `b2` bigint DEFAULT NULL,
  `b3` bigint DEFAULT NULL,
  `i1` bigint DEFAULT NULL,
  `i2` bigint DEFAULT NULL,
  `i3` bigint DEFAULT NULL,
  `r1` double DEFAULT NULL,
  `r2` double DEFAULT NULL,
  `r3` double DEFAULT NULL,
  `c1` text,
  `c2` text,
  `c3` text,
  KEY `unique_ID` (`unique_ID`) USING BTREE,
  KEY `track_id` (`track_id`) USING BTREE,
  KEY `timestamp` (`timestamp`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `YoloTracks`
--

DROP TABLE IF EXISTS `YoloTracks`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `YoloTracks` (
  `time` double DEFAULT NULL,
  `frame_id` bigint DEFAULT NULL,
  `track_id` bigint DEFAULT NULL,
  `center_x` double DEFAULT NULL,
  `center_y` double DEFAULT NULL,
  `class` text,
  `timestamp` datetime(6) DEFAULT NULL,
  `intersection_id` bigint DEFAULT NULL,
  `camera_id` bigint DEFAULT NULL,
  `nearmiss` bigint DEFAULT NULL,
  `signature` bigint DEFAULT NULL,
  `unique_ID` bigint DEFAULT NULL,
  `raw_x` double DEFAULT NULL,
  `raw_y` double DEFAULT NULL,
  `b1` bigint DEFAULT NULL,
  `b2` bigint DEFAULT NULL,
  `b3` bigint DEFAULT NULL,
  `i1` bigint DEFAULT NULL,
  `i2` bigint DEFAULT NULL,
  `i3` bigint DEFAULT NULL,
  `r1` double DEFAULT NULL,
  `r2` double DEFAULT NULL,
  `r3` double DEFAULT NULL,
  `c1` text,
  `c2` text,
  `c3` text,
  KEY `unique_ID` (`unique_ID`) USING BTREE,
  KEY `track_id` (`track_id`) USING BTREE,
  KEY `timestamp` (`timestamp`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `bluetoad`
--

DROP TABLE IF EXISTS `bluetoad`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `bluetoad` (
  `route_id` varchar(20) DEFAULT NULL,
  `day` varchar(10) DEFAULT NULL,
  `timestamp` datetime DEFAULT NULL,
  `lastmatch` datetime DEFAULT NULL,
  `travel_time` float DEFAULT NULL,
  `speed` float DEFAULT NULL,
  `12wk_speed_avg` float DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `gapstats`
--

DROP TABLE IF EXISTS `gapstats`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `gapstats` (
  `unique_ID` bigint DEFAULT NULL,
  `center_x` varchar(45) DEFAULT NULL,
  `center_y` varchar(45) DEFAULT NULL,
  `frame_id` varchar(45) DEFAULT NULL,
  `timestamp` varchar(45) DEFAULT NULL,
  `inst_speed` varchar(45) DEFAULT NULL,
  `absspeed` varchar(45) DEFAULT NULL,
  `speeddiff` varchar(45) DEFAULT NULL,
  `accleration` varchar(45) DEFAULT NULL,
  `gap_conflict_dis` varchar(45) DEFAULT NULL,
  `gap_conflict_PET` varchar(45) DEFAULT NULL,
  `gap_conflict_TTC` varchar(45) DEFAULT NULL,
  `gap_conflict_acc` varchar(45) DEFAULT NULL,
  `speed_x` varchar(45) DEFAULT NULL,
  `speed_y` varchar(45) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `loginData`
--

DROP TABLE IF EXISTS `loginData`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `loginData` (
  `email` varchar(100) DEFAULT NULL,
  `password` varchar(100) DEFAULT NULL,
  KEY `email` (`email`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `new_ttctable`
--

DROP TABLE IF EXISTS `new_ttctable`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `new_ttctable` (
  `intersection_id` int DEFAULT NULL,
  `camera_id` int DEFAULT NULL,
  `timestamp` datetime(6) DEFAULT NULL,
  `dow` int DEFAULT NULL,
  `hod` int DEFAULT NULL,
  `frame_id` int DEFAULT NULL,
  `conflict_x` float DEFAULT NULL,
  `conflict_y` float DEFAULT NULL,
  `unique_ID1` bigint DEFAULT NULL,
  `unique_ID2` bigint DEFAULT NULL,
  `class1` varchar(20) DEFAULT NULL,
  `class2` varchar(20) DEFAULT NULL,
  `phase1` int DEFAULT NULL,
  `phase2` int DEFAULT NULL,
  `time` float DEFAULT NULL,
  `bb_time` float DEFAULT NULL,
  `ttc_rank` float DEFAULT NULL,
  `p2v` tinyint(1) DEFAULT NULL,
  `city` varchar(20) DEFAULT NULL,
  `state` varchar(10) DEFAULT NULL,
  `cluster1` varchar(30) DEFAULT NULL,
  `cluster2` varchar(30) DEFAULT NULL,
  `is_conflicting` float DEFAULT NULL,
  `speed1` float DEFAULT NULL,
  `speed2` float DEFAULT NULL,
  `distance` float DEFAULT NULL,
  `bb_distance` float DEFAULT NULL,
  `deceleration1` float DEFAULT NULL,
  `deceleration2` float DEFAULT NULL,
  `decel1_ts` datetime(6) DEFAULT NULL,
  `decel2_ts` datetime(6) DEFAULT NULL,
  `type` int DEFAULT NULL,
  `signal_state` varchar(20) DEFAULT NULL,
  `phase_duration` float DEFAULT NULL,
  `percent_in_phase` float DEFAULT NULL,
  `ped_signal_state` varchar(20) DEFAULT NULL,
  `ped_phase_duration` float DEFAULT NULL,
  `ped_percent_in_phase` float DEFAULT NULL,
  `cpi1` float DEFAULT NULL,
  `cpi2` float DEFAULT NULL,
  `angle` int DEFAULT NULL,
  `cycle` int DEFAULT NULL,
  `num_involved` int DEFAULT NULL,
  `intersection_diagonal` float DEFAULT NULL,
  `median_width` float DEFAULT NULL,
  `total_vehicles` int DEFAULT NULL,
  `conflict_type` int DEFAULT '0',
  `include_flag` int DEFAULT '0'
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `testATSPM`
--

DROP TABLE IF EXISTS `testATSPM`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `testATSPM` (
  `intersection_id` int DEFAULT NULL,
  `timestamp` datetime(3) DEFAULT NULL,
  `eventCode` int DEFAULT NULL,
  `eventParam` int DEFAULT NULL,
  KEY `timestamp` (`timestamp`) USING BTREE,
  KEY `eventCode` (`eventCode`) USING BTREE,
  KEY `intersection_id` (`intersection_id`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `test_table`
--

DROP TABLE IF EXISTS `test_table`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `test_table` (
  `id` int DEFAULT NULL,
  `value` varchar(20) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `tracksReal`
--

DROP TABLE IF EXISTS `tracksReal`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `tracksReal` (
  `track_id` bigint DEFAULT NULL,
  `Class` text,
  `Approach` text,
  `min_gap` double DEFAULT NULL,
  `max_gap` double DEFAULT NULL,
  `avg_gap` double DEFAULT NULL,
  `Responsetime` double DEFAULT NULL,
  `average_speed` double DEFAULT NULL,
  `Max_speed` double DEFAULT NULL,
  `Min_speed` double DEFAULT NULL,
  `Cluster` text,
  `Cycle` bigint DEFAULT NULL,
  `start_phase` text,
  `end_phase` text,
  `start_timestamp` datetime DEFAULT NULL,
  `end_timestamp` datetime DEFAULT NULL,
  `redJump` bigint DEFAULT NULL,
  `trackTurn` bigint DEFAULT NULL,
  `intersection_id` bigint DEFAULT NULL,
  `camera_id` bigint DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `ttc_display`
--

DROP TABLE IF EXISTS `ttc_display`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `ttc_display` (
  `phase1` int DEFAULT NULL,
  `x1` int DEFAULT NULL,
  `y1` int DEFAULT NULL,
  `ttc1x` decimal(32,0) DEFAULT NULL,
  `intersection_id` int DEFAULT NULL,
  `threshold` int DEFAULT NULL,
  `timestamp1` datetime DEFAULT NULL,
  `value` decimal(32,0) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `tti_all_tracks`
--

DROP TABLE IF EXISTS `tti_all_tracks`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `tti_all_tracks` (
  `frame_id` int DEFAULT NULL,
  `unique_ID1` varchar(45) DEFAULT NULL,
  `unique_ID2` varchar(45) DEFAULT NULL,
  `gap_conflict_dis` decimal(64,0) DEFAULT NULL,
  `x1` decimal(64,0) DEFAULT NULL,
  `y1` decimal(64,0) DEFAULT NULL,
  `phase1` int DEFAULT NULL,
  `ttc1x` decimal(64,0) DEFAULT NULL,
  `ttc1y` decimal(64,0) DEFAULT NULL,
  `x2` decimal(64,0) DEFAULT NULL,
  `y2` decimal(64,0) DEFAULT NULL,
  `ttc2x` decimal(64,0) DEFAULT NULL,
  `ttc2y` decimal(64,0) DEFAULT NULL,
  `speedx1` decimal(64,0) DEFAULT NULL,
  `speedy1` decimal(64,0) DEFAULT NULL,
  `conflict_pointx` decimal(64,0) DEFAULT NULL,
  `conflict_pointy` decimal(64,0) DEFAULT NULL,
  `speedx2` decimal(64,0) DEFAULT NULL,
  `speedy2` decimal(64,0) DEFAULT NULL,
  `phase2` decimal(64,0) DEFAULT NULL,
  `timestamp1` varchar(65) DEFAULT NULL,
  `timestamp2` varchar(65) DEFAULT NULL,
  `intersection_id` int DEFAULT NULL,
  `threshold` int DEFAULT NULL,
  `v2v` int DEFAULT NULL,
  KEY `idx_tti_all_tracks_timestamp1` (`timestamp1`),
  KEY `idx_tti_all_tracks_intersection_id` (`intersection_id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping events for database 'testdb'
--

--
-- Dumping routines for database 'testdb'
--
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2025-07-31 14:35:49
